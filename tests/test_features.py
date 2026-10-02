"""Engineered columns, including nulls, and genre indicators."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.clean import clean_catalogue
from src.features.build_pipeline import GenreBinarizer
from src.features.engineer import engineer_features


def test_engineered_numbers_on_a_hand_built_row():
    config = load_config()
    description = "A family faces murder and drug crime in school"
    title = "Amélie, A Tale"
    frame = pd.DataFrame(
        [
            {
                "show_id": "s1",
                "type": "TV Show",
                "title": title,
                "director": "Raúl Campos, Jan Suter",
                "cast": "A, B, C",
                "country": "United States, India",
                "date_added": "September 25, 2021",
                "release_year": 2018,
                "rating": "TV-Y",
                "duration": "2 Seasons",
                "listed_in": "Kids' TV, Dramas, Comedies, Thrillers, Documentaries",
                "description": description,
            }
        ]
    )
    features = engineer_features(frame, config).iloc[0]
    words = description.split()
    assert features["n_directors"] == 2
    assert features["has_director"] == 1
    assert features["n_cast"] == 3
    assert features["n_countries"] == 2
    assert features["primary_country"] == "United States"
    assert features["primary_genre"] == "Kids' TV"
    assert features["n_genres"] == 5
    assert math.isnan(features["duration_minutes"])
    assert features["duration_seasons"] == 2
    assert features["year_added"] == 2021
    assert features["month_added"] == 9
    assert features["years_to_platform"] == 3
    assert features["description_char_len"] == len(description)
    assert features["description_word_count"] == 9
    assert features["mean_word_length"] == pytest_mean(words)
    assert features["title_word_count"] == 3
    assert features["lex_violence"] == 2
    assert features["lex_substance"] == 1
    assert features["lex_family"] == 2
    assert features["lex_sexual"] == 0


def test_null_director_cast_and_country_become_zeros():
    config = load_config()
    frame = pd.DataFrame(
        [
            {
                "show_id": "s1",
                "type": "Movie",
                "title": "Solo",
                "director": "",
                "cast": "",
                "country": "",
                "date_added": "",
                "release_year": 2020,
                "rating": "PG",
                "duration": "90 min",
                "listed_in": "Dramas",
                "description": "",
            }
        ]
    )
    features = engineer_features(frame, config).iloc[0]
    assert features["has_director"] == 0
    assert features["n_directors"] == 0
    assert features["n_cast"] == 0
    assert features["n_countries"] == 0
    assert features["primary_country"] == ""
    assert features["description_word_count"] == 0
    assert features["mean_word_length"] == 0
    assert math.isnan(features["year_added"])
    assert math.isnan(features["years_to_platform"])
    assert features["duration_minutes"] == 90
    assert math.isnan(features["duration_seasons"])


def test_not_given_is_not_a_director_after_cleaning():
    config = load_config()
    raw = pd.DataFrame(
        [
            {
                "show_id": "s1",
                "type": "Movie",
                "title": "Solo",
                "director": "Not Given",
                "cast": "",
                "country": "Not Given",
                "date_added": "9/25/2021",
                "release_year": "2020",
                "rating": "PG",
                "duration": "90 min",
                "listed_in": "Dramas",
                "description": "",
            }
        ]
    )
    cleaned = clean_catalogue(raw, config)
    features = engineer_features(cleaned, config).iloc[0]
    assert features["has_director"] == 0
    assert features["n_countries"] == 0
    assert features["year_added"] == 2021
    assert features["month_added"] == 9


def test_genre_binarizer_handles_one_genre_and_an_unseen_genre():
    encoder = GenreBinarizer()
    train = pd.DataFrame({"listed_in": ["Dramas, Comedies", "Dramas"]})
    encoder.fit(train)
    transformed = encoder.transform(pd.DataFrame({"listed_in": ["Dramas", "Brand New Genre"]}))
    assert transformed.shape == (2, 2)
    assert transformed.sum(axis=1).tolist() == [1, 0]
    names = set(encoder.get_feature_names_out())
    assert names == {"genre__Comedies", "genre__Dramas"}


def test_indicator_columns_mark_missing_numeric_values():
    from sklearn.impute import SimpleImputer

    values = pd.DataFrame({"duration_minutes": [90.0, np.nan], "duration_seasons": [np.nan, 2.0]})
    imputed = SimpleImputer(strategy="median", add_indicator=True).fit_transform(values)
    assert imputed.shape[1] == 4
    assert imputed[1, 2] == 1
    assert imputed[0, 3] == 1


def pytest_mean(words: list[str]) -> float:
    return sum(len(word) for word in words) / len(words)
