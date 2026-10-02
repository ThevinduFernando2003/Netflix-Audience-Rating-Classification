"""Row-local features. Nothing here is fitted, so it is safe to run before the split."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.config import REPO_ROOT
from src.data.clean import is_missing_text, parse_date_added, parse_duration
from src.features.lexicon import lexicon_counts, load_lexicons

TEXT_COLUMNS: list[str] = ["title", "description", "listed_in"]
CATEGORICAL_COLUMNS: list[str] = ["type", "primary_country", "primary_genre"]
NUMERIC_COLUMNS: list[str] = [
    "release_year",
    "year_added",
    "month_added",
    "years_to_platform",
    "duration_minutes",
    "duration_seasons",
    "n_genres",
    "n_cast",
    "n_countries",
    "n_directors",
    "description_char_len",
    "description_word_count",
    "mean_word_length",
    "title_char_len",
    "title_word_count",
    "title_mean_word_length",
    "has_director",
    "lex_violence",
    "lex_sexual",
    "lex_substance",
    "lex_family",
]


def split_multi(value: object, sentinels: set[str]) -> list[str]:
    """Split a comma-separated cell, strip each part, and drop empties and sentinels."""
    if is_missing_text(value):
        return []
    text = str(value).strip()
    if text.lower() in sentinels:
        return []
    parts: list[str] = []
    for part in text.split(","):
        item = part.strip()
        if item and item.lower() not in sentinels:
            parts.append(item)
    return parts


def _word_stats(text: str) -> tuple[int, int, float]:
    words = text.split()
    if not words:
        return 0, 0, 0.0
    return len(text), len(words), float(sum(len(word) for word in words) / len(words))


def engineer_features(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Build the model frame from a cleaned catalogue frame.

    `description` and `cast` are real columns even when the source file omitted them
    (they are empty). Length features of an empty description are 0, which is the
    measurement, not an imputed word like Unknown.
    """
    sentinels = {token.lower() for token in config["missing_sentinels"]}
    lexicons = load_lexicons(config["lexicons"], REPO_ROOT)
    rows: list[dict[str, object]] = []
    for record in frame.itertuples(index=False):
        title = "" if is_missing_text(record.title) else str(record.title).strip()
        description = "" if is_missing_text(record.description) else str(record.description).strip()
        listed = split_multi(record.listed_in, sentinels)
        directors = split_multi(record.director, sentinels)
        cast = split_multi(record.cast, sentinels)
        countries = split_multi(record.country, sentinels)
        added = parse_date_added(record.date_added, config["date_formats"])
        if "duration_minutes" in frame.columns and not is_missing_text(getattr(record, "duration_minutes", None)):
            minutes = float(record.duration_minutes) if not _is_nan(record.duration_minutes) else np.nan
            seasons = float(record.duration_seasons) if not _is_nan(record.duration_seasons) else np.nan
        else:
            minutes, seasons = parse_duration(record.duration)
        year_added = float(added.year) if not pd.isna(added) else np.nan
        month_added = float(added.month) if not pd.isna(added) else np.nan
        release_year = float(record.release_year) if not _is_nan(record.release_year) else np.nan
        years_to_platform = (
            year_added - release_year if not math.isnan(year_added) and not math.isnan(release_year) else np.nan
        )
        desc_chars, desc_words, desc_mean = _word_stats(description)
        title_chars, title_words, title_mean = _word_stats(title)
        counts = lexicon_counts(f"{title} {description}", lexicons)
        rows.append(
            {
                "show_id": str(record.show_id),
                "title": title,
                "description": description,
                "listed_in": ", ".join(listed),
                "type": str(record.type),
                "primary_country": countries[0] if countries else "",
                "primary_genre": listed[0] if listed else "",
                "release_year": release_year,
                "year_added": year_added,
                "month_added": month_added,
                "years_to_platform": years_to_platform,
                "duration_minutes": minutes,
                "duration_seasons": seasons,
                "n_genres": float(len(listed)),
                "n_cast": float(len(cast)),
                "n_countries": float(len(countries)),
                "n_directors": float(len(directors)),
                "description_char_len": float(desc_chars),
                "description_word_count": float(desc_words),
                "mean_word_length": float(desc_mean),
                "title_char_len": float(title_chars),
                "title_word_count": float(title_words),
                "title_mean_word_length": float(title_mean),
                "has_director": float(len(directors) > 0),
                "lex_violence": float(counts["violence"]),
                "lex_sexual": float(counts["sexual"]),
                "lex_substance": float(counts["substance"]),
                "lex_family": float(counts["family"]),
            }
        )
    return pd.DataFrame(rows)


def _is_nan(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        return bool(math.isnan(float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def description_has_text(frame: pd.DataFrame) -> bool:
    if "description" not in frame.columns:
        return False
    return bool(frame["description"].fillna("").astype(str).str.strip().ne("").any())
