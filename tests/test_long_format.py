"""Tests for the long-format reshape helper and feature metadata table.

Covers:
- `to_long_format` melt/round-trip correctness
- `join_metadata` behavior, including the fallback for feature names that
  don't have an exact metadata match (dynamic IDyOM columns)
- `get_feature_metadata` basic shape/consistency
- End-to-end `get_all_features(..., long_format=True)`
"""

import numpy as np
import pandas as pd
import pytest

from melody_features.core.representations import Melody
from melody_features.feature_metadata import get_feature_metadata
from melody_features.features import get_all_features
from melody_features.reshape import to_long_format

# Families that are always produced by decorator-typed features (i.e. not
# corpus, which needs a configured corpus, and not idyom, which is dynamic).
STATIC_FAMILIES = {
    "absolute_pitch",
    "pitch_class",
    "pitch_interval",
    "contour",
    "timing",
    "inter_onset_interval",
    "tonality",
    "metre",
    "expectation",
    "complexity",
    "lexical_diversity",
}


@pytest.fixture(scope="module")
def wide_df():
    melodies = [
        Melody.from_notes(
            [60, 62, 64, 65, 67, 69, 71, 72],
            [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            [0.4, 0.9, 1.4, 1.9, 2.4, 2.9, 3.4, 3.9],
        ),
        Melody.from_notes(
            [72, 71, 69, 67, 65, 64, 62, 60],
            [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            [0.4, 0.9, 1.4, 1.9, 2.4, 2.9, 3.4, 3.9],
        ),
    ]
    return get_all_features(melodies, skip_idyom=True)


class TestToLongFormatBasics:
    def test_melt_shape_and_id_columns(self):
        wide = pd.DataFrame(
            {
                "melody_num": [1, 2],
                "melody_id": ["a", "b"],
                "absolute_pitch.pitch_range": [10, 20],
                "timing.note_density": [1.5, 2.5],
            }
        )
        long_df = to_long_format(wide, join_metadata=False)

        assert len(long_df) == 2 * 2  # n_melodies * n_feature_columns
        assert set(long_df.columns) == {"melody_num", "melody_id", "feature_name", "value"}
        assert set(long_df["feature_name"]) == {"absolute_pitch.pitch_range", "timing.note_density"}
        assert set(long_df.loc[long_df["feature_name"] == "absolute_pitch.pitch_range", "value"]) == {10, 20}

    def test_round_trip_pivot_matches_original(self):
        wide = pd.DataFrame(
            {
                "melody_num": [1, 2],
                "melody_id": ["a", "b"],
                "absolute_pitch.pitch_range": [10, 20],
                "contour.polynomial_contour_coefficients": [[1.0, 2.0], [3.0, 4.0]],
            }
        )
        long_df = to_long_format(wide, join_metadata=False)
        pivoted = (
            long_df.pivot(index=["melody_num", "melody_id"], columns="feature_name", values="value")
            .reset_index()[wide.columns]
        )

        assert pivoted.loc[0, "absolute_pitch.pitch_range"] == 10
        assert pivoted.loc[1, "absolute_pitch.pitch_range"] == 20
        assert list(pivoted.loc[0, "contour.polynomial_contour_coefficients"]) == [1.0, 2.0]
        assert list(pivoted.loc[1, "contour.polynomial_contour_coefficients"]) == [3.0, 4.0]

    def test_preserves_list_and_dict_valued_features(self):
        wide = pd.DataFrame(
            {
                "melody_num": [1],
                "melody_id": ["a"],
                "absolute_pitch.basic_pitch_histogram": [{60: 1, 62: 1}],
                "contour.comb_contour_matrix": [np.array([[1, 2], [3, 4]])],
            }
        )
        long_df = to_long_format(wide, join_metadata=False)
        hist_value = long_df.loc[long_df["feature_name"] == "absolute_pitch.basic_pitch_histogram", "value"].iloc[0]
        assert hist_value == {60: 1, 62: 1}


class TestJoinMetadata:
    def test_join_metadata_adds_expected_columns(self):
        wide = pd.DataFrame(
            {
                "melody_num": [1],
                "melody_id": ["a"],
                "absolute_pitch.pitch_range": [10],
            }
        )
        long_df = to_long_format(wide, join_metadata=True)

        assert {"family", "source", "domain", "type", "description", "notes", "references"} <= set(long_df.columns)
        row = long_df.iloc[0]
        assert row["family"] == "absolute_pitch"
        assert "jSymbolic" in row["source"] or "MIDI Toolbox" in row["source"]
        assert row["domain"] == "pitch"
        assert row["type"] == "Descriptor"

    def test_unmatched_idyom_column_falls_back_gracefully(self):
        wide = pd.DataFrame(
            {
                "melody_num": [1],
                "melody_id": ["a"],
                "idyom.pitch_stm_mean_information_content": [1.23],
            }
        )
        long_df = to_long_format(wide, join_metadata=True)

        row = long_df.iloc[0]
        assert row["family"] == "idyom"
        assert row["source"] == "IDyOM"
        # No exact static metadata for a dynamic per-config IDyOM name.
        assert pd.isna(row["type"])

    def test_unmatched_generic_column_falls_back_to_prefix(self):
        wide = pd.DataFrame(
            {
                "melody_num": [1],
                "melody_id": ["a"],
                "some_unknown_family.some_unknown_feature": [1],
            }
        )
        long_df = to_long_format(wide, join_metadata=True)

        row = long_df.iloc[0]
        assert row["family"] == "some_unknown_family"
        assert not row["family"] != row["family"]  # not NaN

    def test_join_metadata_false_omits_metadata_columns(self):
        wide = pd.DataFrame({"melody_num": [1], "melody_id": ["a"], "absolute_pitch.pitch_range": [10]})
        long_df = to_long_format(wide, join_metadata=False)
        assert set(long_df.columns) == {"melody_num", "melody_id", "feature_name", "value"}

    def test_custom_metadata_table_is_used_when_provided(self):
        wide = pd.DataFrame({"melody_num": [1], "melody_id": ["a"], "family.feature": [1]})
        custom_metadata = pd.DataFrame(
            {
                "feature_name": ["family.feature"],
                "family": ["family"],
                "source": ["Custom"],
            }
        )
        long_df = to_long_format(wide, join_metadata=True, metadata=custom_metadata)
        assert long_df.iloc[0]["source"] == "Custom"


class TestGetFeatureMetadata:
    def test_returns_expected_columns(self):
        meta = get_feature_metadata()
        assert {"feature_name", "family", "source", "domain", "type", "description"} <= set(meta.columns)

    def test_feature_names_are_unique(self):
        meta = get_feature_metadata()
        assert not meta["feature_name"].duplicated().any()

    def test_covers_real_wide_format_columns(self, wide_df):
        meta = get_feature_metadata()
        known = set(meta["feature_name"])
        wide_cols = [c for c in wide_df.columns if c not in ("melody_num", "melody_id")]
        missing = [c for c in wide_cols if c not in known]
        assert not missing, f"Wide columns missing metadata: {missing}"

    def test_static_families_have_no_missing_source(self):
        meta = get_feature_metadata()
        static_rows = meta[meta["family"].isin(STATIC_FAMILIES)]
        assert not static_rows["source"].isna().any()
        assert not (static_rows["source"] == "").any()


class TestGetAllFeaturesLongFormat:
    def test_long_format_matches_to_long_format_of_wide(self, wide_df):
        long_from_flag = to_long_format(wide_df.copy())
        # Simulate calling get_all_features(..., long_format=True) by
        # reshaping the same wide result directly, since re-running
        # get_all_features would duplicate the (slow) feature extraction.
        assert set(long_from_flag["feature_name"]) == {
            c for c in wide_df.columns if c not in ("melody_num", "melody_id")
        }
        assert len(long_from_flag) == len(wide_df) * (len(wide_df.columns) - 2)

    def test_get_all_features_long_format_true_end_to_end(self):
        melodies = [
            Melody.from_notes([60, 62, 64, 65], [0.0, 0.5, 1.0, 1.5], [0.4, 0.9, 1.4, 1.9]),
        ]

        wide = get_all_features(melodies, skip_idyom=True)
        long_direct = get_all_features(melodies, skip_idyom=True, long_format=True)
        long_reshaped = to_long_format(wide)

        assert list(long_direct.columns) == list(long_reshaped.columns)
        assert len(long_direct) == len(long_reshaped)
        assert set(long_direct["feature_name"]) == set(long_reshaped["feature_name"])

    def test_get_all_features_long_format_join_metadata_false(self):
        melodies = [
            Melody.from_notes([60, 62, 64, 65], [0.0, 0.5, 1.0, 1.5], [0.4, 0.9, 1.4, 1.9]),
        ]

        long_df = get_all_features(
            melodies, skip_idyom=True, long_format=True, join_metadata=False
        )
        assert set(long_df.columns) == {"melody_num", "melody_id", "feature_name", "value"}

    def test_default_call_still_returns_wide_format(self, wide_df):
        assert "feature_name" not in wide_df.columns
        assert "melody_num" in wide_df.columns
        assert len(wide_df) == 2

    def test_metric_hierarchy_and_meter_accent_not_duplicated(self, wide_df):
        # metric_hierarchy/meter_accent are dual-tagged (@rhythm and @metre)
        # but must only be materialized once, under `metre.*`; previously
        # they were also duplicated under `timing.*` with identical values.
        assert "metre.metric_hierarchy" in wide_df.columns
        assert "metre.meter_accent" in wide_df.columns
        assert "timing.metric_hierarchy" not in wide_df.columns
        assert "timing.meter_accent" not in wide_df.columns
