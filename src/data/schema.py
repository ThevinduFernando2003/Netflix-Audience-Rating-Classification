"""Structural checks on the catalogue file. Violations raise SchemaError."""

from __future__ import annotations

from typing import Any

import pandas as pd


class SchemaError(ValueError):
    """The catalogue frame does not satisfy the data contract."""


def assert_catalogue_schema(frame: pd.DataFrame, config: dict[str, Any]) -> None:
    """Raise SchemaError naming the broken contract, before any modelling."""
    required = list(config["required_columns"])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SchemaError(
            "Catalogue is missing required columns: "
            + ", ".join(missing)
            + ". Expected at least: "
            + ", ".join(required)
            + ". Optional columns (filled with an empty string when absent): "
            + ", ".join(config["optional_columns"])
            + "."
        )
    if frame.empty:
        raise SchemaError("Catalogue has no rows.")
    if frame["show_id"].isna().any() or frame["show_id"].astype(str).str.strip().eq("").any():
        raise SchemaError("Every row must have a non-empty show_id.")
    duplicated = frame["show_id"].astype(str)[frame["show_id"].astype(str).duplicated()].unique()
    if len(duplicated):
        sample = ", ".join(map(str, duplicated[:10]))
        raise SchemaError(f"Duplicate show_id values: {sample}.")
    allowed = set(config["allowed_types"])
    observed = set(frame["type"].dropna().astype(str).unique())
    unexpected = observed - allowed
    if unexpected:
        raise SchemaError(
            "Unexpected type values "
            + ", ".join(sorted(unexpected))
            + ". Allowed: "
            + ", ".join(sorted(allowed))
            + "."
        )
    if frame["type"].isna().any() or frame["type"].astype(str).str.strip().eq("").any():
        raise SchemaError("Every row must have a type of Movie or TV Show.")
