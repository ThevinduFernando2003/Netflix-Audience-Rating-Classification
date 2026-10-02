"""Defect repair. Contaminated and missing ratings are flagged, never coerced into a class."""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DURATION_MINUTES = re.compile(r"^(\d+)\s*min$", re.IGNORECASE)
DURATION_SEASONS = re.compile(r"^(\d+)\s*seasons?$", re.IGNORECASE)


def contamination_pattern(config: dict[str, Any]) -> re.Pattern[str]:
    return re.compile(config["rating_contamination_pattern"], re.IGNORECASE)


def is_contaminated_rating(value: object, config: dict[str, Any]) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return False
    return contamination_pattern(config).fullmatch(text) is not None


def is_missing_text(value: object) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def apply_sentinels(value: object, sentinels: set[str]) -> str:
    """Return a stripped string, or '' when the cell is a missing sentinel."""
    if is_missing_text(value):
        return ""
    text = str(value).strip()
    if text.lower() in sentinels:
        return ""
    return text


def parse_date_added(value: object, formats: list[str]) -> pd.Timestamp:
    """Parse one date with an explicit format list. Unparseable values become NaT."""
    if is_missing_text(value):
        return pd.NaT
    text = str(value).strip()
    for fmt in formats:
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        if not pd.isna(parsed):
            return pd.Timestamp(parsed)
    return pd.NaT


def parse_duration(value: object) -> tuple[float, float]:
    """Return (minutes, seasons). The unused unit is NaN. Failures are (NaN, NaN)."""
    if is_missing_text(value):
        return (np.nan, np.nan)
    text = str(value).strip()
    minutes = DURATION_MINUTES.fullmatch(text)
    if minutes:
        return (float(minutes.group(1)), np.nan)
    seasons = DURATION_SEASONS.fullmatch(text)
    if seasons:
        return (np.nan, float(seasons.group(1)))
    return (np.nan, np.nan)


def clean_catalogue(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Repair known defects and mark rows that must not enter a training fold.

    A duration string in `rating` is moved into `duration` when duration is empty,
    and `rating` is set to missing. The row stays in the frame so the UI can show
    it, but `include_in_training` is False.
    """
    sentinels = {token.lower() for token in config["missing_sentinels"]}
    cleaned = frame.copy()
    text_columns = ["title", "director", "cast", "country", "listed_in", "description", "type"]
    for column in text_columns:
        cleaned[column] = cleaned[column].map(lambda value: apply_sentinels(value, sentinels))

    reasons: list[str] = []
    ratings: list[str] = []
    durations: list[str] = []
    contaminated_flags: list[bool] = []
    missing_flags: list[bool] = []

    for rating, duration in zip(cleaned["rating"], cleaned["duration"], strict=True):
        rating_text = "" if is_missing_text(rating) else str(rating).strip()
        duration_text = "" if is_missing_text(duration) else str(duration).strip()
        contaminated = is_contaminated_rating(rating_text, config)
        if contaminated:
            if duration_text == "":
                duration_text = rating_text
            rating_text = ""
            reasons.append("rating_contaminated")
            contaminated_flags.append(True)
            missing_flags.append(False)
        elif rating_text == "":
            reasons.append("rating_missing")
            contaminated_flags.append(False)
            missing_flags.append(True)
        else:
            reasons.append("")
            contaminated_flags.append(False)
            missing_flags.append(False)
        ratings.append(rating_text)
        durations.append(duration_text)

    cleaned["rating"] = ratings
    cleaned["duration"] = durations
    cleaned["exclude_reason"] = reasons
    cleaned["rating_contaminated"] = contaminated_flags
    cleaned["rating_missing"] = missing_flags
    cleaned["include_in_training"] = [reason == "" for reason in reasons]

    date_parsed = cleaned["date_added"].map(lambda value: parse_date_added(value, config["date_formats"]))
    raw_date_present = ~cleaned["date_added"].map(is_missing_text)
    cleaned["date_parse_failed"] = raw_date_present & date_parsed.isna()

    parsed_duration = cleaned["duration"].map(parse_duration)
    cleaned["duration_minutes"] = [item[0] for item in parsed_duration]
    cleaned["duration_seasons"] = [item[1] for item in parsed_duration]
    raw_duration_present = ~cleaned["duration"].map(is_missing_text)
    cleaned["duration_parse_failed"] = raw_duration_present & cleaned["duration_minutes"].isna() & cleaned[
        "duration_seasons"
    ].isna()

    cleaned["release_year"] = pd.to_numeric(cleaned["release_year"], errors="coerce")

    n_contaminated = int(cleaned["rating_contaminated"].sum())
    n_missing = int(cleaned["rating_missing"].sum())
    n_date_fail = int(cleaned["date_parse_failed"].sum())
    n_duration_fail = int(cleaned["duration_parse_failed"].sum())
    logger.info(
        "Cleaned %s rows: contaminated_rating=%s missing_rating=%s date_parse_failures=%s duration_parse_failures=%s",
        len(cleaned),
        n_contaminated,
        n_missing,
        n_date_fail,
        n_duration_fail,
    )
    return cleaned
