"""Reshape helpers for the wide-format DataFrame returned by `get_all_features`."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .feature_metadata import get_feature_metadata

ID_VARS = ["melody_num", "melody_id"]
_METADATA_VALUE_COLUMNS = ["family", "source", "domain", "type", "description", "notes", "references"]


def _fallback_family(feature_name: str) -> str:
    """Infer a family from a `feature_name` that has no exact metadata match.

    This chiefly covers dynamically-named IDyOM columns (`idyom.<config>_<metric>`)
    and the handful of legacy columns that `get_all_features` computes twice
    under two family prefixes for backward compatibility (see
    `feature_metadata.get_feature_metadata`'s docstring).
    """
    return feature_name.split(".", 1)[0] if "." in feature_name else feature_name


def to_long_format(
    df: pd.DataFrame,
    *,
    join_metadata: bool = True,
    metadata: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Reshape a wide-format feature DataFrame into tidy long format.

    Parameters
    ----------
    df : pd.DataFrame
        A wide-format DataFrame as returned by `get_all_features`, with
        `melody_num`/`melody_id` identifier columns and one `{family}.{feature}`
        column per feature.
    join_metadata : bool, optional
        If True (default), left-join feature metadata (family, source,
        domain, type, description, notes, references) onto the long
        DataFrame by `feature_name`. Any `feature_name` without an exact
        metadata match (chiefly dynamic IDyOM columns) falls back to a
        family/source inferred from the column prefix rather than being
        left blank.
    metadata : pd.DataFrame, optional
        A metadata table to join instead of the default
        :func:`melody_features.get_feature_metadata` table (for example, a
        filtered or user-extended version). Must contain a `feature_name`
        column plus any of the columns in
        `family, source, domain, type, description, notes, references`.

    Returns
    -------
    pd.DataFrame
        Columns: `melody_num`, `melody_id`, `feature_name`, `value`, and
        (when `join_metadata=True`) `family`, `source`, `domain`, `type`,
        `description`, `notes`, `references`.
    """
    id_vars = [col for col in ID_VARS if col in df.columns]
    value_vars = [col for col in df.columns if col not in id_vars]

    long_df = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="feature_name",
        value_name="value",
    )
    # Keep heterogeneous feature values (scalars, lists, dicts) intact rather
    # than letting pandas upcast everything to a common dtype.
    long_df["value"] = long_df["value"].astype(object)

    if not join_metadata:
        return long_df

    meta = metadata if metadata is not None else get_feature_metadata()
    meta_cols = ["feature_name"] + [c for c in _METADATA_VALUE_COLUMNS if c in meta.columns]
    long_df = long_df.merge(meta[meta_cols], on="feature_name", how="left")

    unmatched = long_df["family"].isna()
    if unmatched.any():
        long_df.loc[unmatched, "family"] = long_df.loc[unmatched, "feature_name"].map(_fallback_family)
        if "source" in long_df.columns:
            is_idyom = unmatched & (long_df["family"] == "idyom")
            long_df.loc[is_idyom, "source"] = "IDyOM"

    trailing_meta_cols = [c for c in ("description", "notes", "references") if c in long_df.columns]
    final_cols = id_vars + ["feature_name", "family", "source", "domain", "type", "value"] + trailing_meta_cols
    return long_df[[c for c in final_cols if c in long_df.columns]]
