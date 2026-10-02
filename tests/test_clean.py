"""Cleaning defects from the data contract."""

from __future__ import annotations

import pandas as pd

from src.config import load_config
from src.data.clean import clean_catalogue
from src.data.labels import training_population
from src.data.load import load_raw_catalogue


def test_contaminated_rating_moves_into_empty_duration(tmp_path):
    config = load_config()
    frame = _one_row(rating="74 min", duration="")
    cleaned = clean_catalogue(frame, config)
    assert cleaned.loc[0, "duration"] == "74 min"
    assert cleaned.loc[0, "rating"] == ""
    assert cleaned.loc[0, "include_in_training"] == False  # noqa: E712
    assert cleaned.loc[0, "exclude_reason"] == "rating_contaminated"
    assert training_population(cleaned).empty


def test_contaminated_rating_does_not_overwrite_existing_duration():
    config = load_config()
    cleaned = clean_catalogue(_one_row(rating="74 min", duration="90 min"), config)
    assert cleaned.loc[0, "duration"] == "90 min"
    assert cleaned.loc[0, "rating"] == ""
    assert bool(cleaned.loc[0, "include_in_training"]) is False


def test_missing_rating_is_excluded_from_training():
    config = load_config()
    cleaned = clean_catalogue(_one_row(rating="", duration="90 min"), config)
    assert cleaned.loc[0, "exclude_reason"] == "rating_missing"
    assert training_population(cleaned).empty


def test_nr_and_ur_stay_in_the_cleaned_frame_but_are_not_buckets():
    config = load_config()
    cleaned = clean_catalogue(pd.concat([_one_row(rating="NR"), _one_row(rating="UR", show_id="s2")], ignore_index=True), config)
    assert set(training_population(cleaned)["rating"]) == {"NR", "UR"}


def test_not_given_is_missing_not_unknown():
    config = load_config()
    cleaned = clean_catalogue(_one_row(director="Not Given", country="Not Given"), config)
    assert cleaned.loc[0, "director"] == ""
    assert cleaned.loc[0, "country"] == ""
    assert "Unknown" not in cleaned.loc[0, "director"]
    assert "Unknown" not in cleaned.loc[0, "country"]


def test_unparseable_date_is_flagged_and_the_row_is_kept():
    config = load_config()
    cleaned = clean_catalogue(_one_row(date_added="not-a-date", rating="PG"), config)
    assert bool(cleaned.loc[0, "date_parse_failed"]) is True
    assert bool(cleaned.loc[0, "include_in_training"]) is True


def test_fixture_training_set_excludes_contaminated_and_missing_ratings():
    config = load_config()
    raw = load_raw_catalogue(_fixture(), config)
    cleaned = clean_catalogue(raw, config)
    trainable = training_population(cleaned)
    assert "74 min" not in set(trainable["rating"])
    assert "" not in set(trainable["rating"])
    assert int((~cleaned["include_in_training"]).sum()) >= 3


def _fixture():
    from pathlib import Path

    return Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mini_netflix.csv"


def _one_row(**overrides: str) -> pd.DataFrame:
    base = {
        "show_id": "s1",
        "type": "Movie",
        "title": "Example",
        "director": "Ada Lovelace",
        "cast": "Actor One",
        "country": "United States",
        "date_added": "September 25, 2021",
        "release_year": "2019",
        "rating": "PG",
        "duration": "90 min",
        "listed_in": "Dramas",
        "description": "A short description.",
    }
    base.update(overrides)
    return pd.DataFrame([base])
