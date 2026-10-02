"""Bucket map, coverage, and rare-class folding."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import REPO_ROOT, load_config
from src.data.labels import (
    LabelMapError,
    assert_ordinal_levels_contiguous,
    assert_ratings_covered,
    bucket_labels,
    known_rating_tokens,
    rating_to_bucket,
    stabilize_raw_labels,
)


def test_every_configured_rating_maps_to_the_expected_bucket():
    config = load_config()
    expected = {
        "TV-Y": ("Kids", 0),
        "TV-G": ("Kids", 0),
        "G": ("Kids", 0),
        "TV-Y7": ("Older Kids", 1),
        "TV-Y7-FV": ("Older Kids", 1),
        "PG": ("Older Kids", 1),
        "TV-PG": ("Older Kids", 1),
        "PG-13": ("Teens", 2),
        "TV-14": ("Teens", 2),
        "R": ("Mature", 3),
        "NC-17": ("Mature", 3),
        "TV-MA": ("Mature", 3),
    }
    assert rating_to_bucket(config) == expected


def test_ordinal_levels_are_contiguous_from_zero():
    assert_ordinal_levels_contiguous(load_config())


def test_nr_and_ur_are_unrated_and_not_on_the_ladder():
    config = load_config()
    mapping = rating_to_bucket(config)
    assert "NR" not in mapping
    assert "UR" not in mapping
    assert known_rating_tokens(config) >= {"NR", "UR", "TV-MA", "G"}
    labels = bucket_labels(pd.Series(["NR", "UR", "TV-MA"]), config)
    assert labels["bucket"].iloc[0] is pd.NA or pd.isna(labels["bucket"].iloc[0])
    assert labels["bucket"].iloc[1] is pd.NA or pd.isna(labels["bucket"].iloc[1])
    assert labels["bucket"].iloc[2] == "Mature"
    assert int(labels["ordinal"].iloc[2]) == 3


def test_unmapped_rating_fails():
    config = load_config()
    with pytest.raises(LabelMapError, match="Unmapped rating"):
        assert_ratings_covered(["TV-MA", "XYZ-NEW"], config)


def test_duration_contamination_is_not_an_unmapped_rating():
    assert_ratings_covered(["TV-MA", "74 min", ""], load_config())


def test_fixture_and_real_catalogue_ratings_are_covered():
    config = load_config()
    fixture = pd.read_csv(Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mini_netflix.csv", dtype=str, keep_default_na=False)
    assert_ratings_covered(fixture["rating"].tolist(), config)
    raw = REPO_ROOT / "data" / "raw" / "netflix_titles.csv"
    if not raw.exists():
        pytest.skip("raw catalogue is not exported")
    catalogue = pd.read_csv(raw, dtype=str, keep_default_na=False)
    assert_ratings_covered(catalogue["rating"].tolist(), config)


def test_rare_classes_fold_into_other_and_tiny_other_is_dropped():
    ratings = pd.Series(["TV-MA"] * 6 + ["NC-17", "NC-17", "UR"])
    labels, report = stabilize_raw_labels(ratings, min_count=5, n_splits=5, other_label="OTHER")
    assert report["folded_into_other"] == {"NC-17": 2, "UR": 1}
    assert report["dropped_other"] == 3
    assert labels.dropna().nunique() == 1
    assert set(labels.dropna()) == {"TV-MA"}
