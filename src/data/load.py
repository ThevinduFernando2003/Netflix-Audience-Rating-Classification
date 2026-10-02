"""Load the raw catalogue and present a stable column set to the rest of the code."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.schema import assert_catalogue_schema

logger = logging.getLogger(__name__)

CANONICAL_COLUMNS: list[str] = [
    "show_id",
    "type",
    "title",
    "director",
    "cast",
    "country",
    "date_added",
    "release_year",
    "rating",
    "duration",
    "listed_in",
    "description",
]


def load_raw_catalogue(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    """Read the CSV. Blank cells stay blank. Optional columns are added as empty strings.

    `keep_default_na=False` so a literal title or name is never swallowed as NA.
    """
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    assert_catalogue_schema(frame, config)
    for column in config["optional_columns"]:
        if column not in frame.columns:
            logger.warning(
                "Column %s is absent from %s. Filling with an empty string. "
                "No model will be allowed to treat that fill as observed text.",
                column,
                path,
            )
            frame[column] = ""
    missing = [column for column in CANONICAL_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Internal column list is incomplete: {missing}")
    ordered = frame.loc[:, CANONICAL_COLUMNS].copy()
    for column in CANONICAL_COLUMNS:
        ordered[column] = ordered[column].map(lambda value: "" if value is None else str(value).strip())
    return ordered
