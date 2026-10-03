"""Build the Formulation B training frame. The sealed test ids are never returned."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import RANDOM_SEED, resolve_path
from src.data.clean import clean_catalogue
from src.data.labels import assert_ratings_covered, rating_to_bucket, training_population
from src.data.load import load_raw_catalogue
from src.data.split import load_sealed_split, save_sealed_split, sealed_split
from src.features.engineer import description_has_text, engineer_features


def _prepare_population(config: dict[str, Any]) -> tuple[pd.DataFrame, set[str]]:
    """Clean the catalogue and load the sealed split. Does not score anything."""
    raw_path = resolve_path(config, "raw_csv")
    raw = load_raw_catalogue(raw_path, config)
    assert_ratings_covered(raw["rating"].tolist(), config)
    cleaned = clean_catalogue(raw, config)
    population = training_population(cleaned)
    mapping = rating_to_bucket(config)
    buckets: list[object] = []
    ordinals: list[float] = []
    for rating in population["rating"].astype(str):
        found = mapping.get(rating)
        if found is None:
            buckets.append(pd.NA)
            ordinals.append(float("nan"))
        else:
            buckets.append(found[0])
            ordinals.append(float(found[1]))
    population = population.copy()
    population["true_bucket"] = buckets
    population["true_ordinal"] = ordinals

    split_path = resolve_path(config, "sealed_split")
    if split_path.exists():
        split = load_sealed_split(split_path)
    else:
        strata = population["true_bucket"].map(lambda value: "UNRATED" if pd.isna(value) else str(value))
        split = sealed_split(
            population["show_id"],
            strata,
            test_size=float(config["split"]["test_size"]),
            seed=RANDOM_SEED,
        )
        split["seed"] = RANDOM_SEED
        split["test_size"] = float(config["split"]["test_size"])
        split["stratify"] = "maturity bucket, with NR/UR as UNRATED"
        save_sealed_split(split, split_path)
    return population, {str(item) for item in split["test_show_ids"]}


def _labelled_frame(subset: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    labelled = subset.loc[subset["true_ordinal"].notna()].copy()
    features = engineer_features(labelled, config)
    frame = features.merge(
        labelled[["show_id", "true_bucket", "true_ordinal"]],
        on="show_id",
        how="left",
        validate="one_to_one",
    )
    frame["y"] = frame["true_bucket"].astype(str)
    return frame.reset_index(drop=True)


def formulation_b_training_frame(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the catalogue, apply the sealed split, and return bucket-labelled training rows.

    The returned frame has engineered columns plus `y` (the bucket name). Test show ids
    are not included.
    """
    population, test_ids = _prepare_population(config)
    train = population.loc[~population["show_id"].astype(str).isin(test_ids)].copy()
    frame = _labelled_frame(train, config)
    leaked = set(frame["show_id"].astype(str)).intersection(test_ids)
    if leaked:
        raise RuntimeError(f"Sealed test show_ids entered the training frame: {len(leaked)}")
    info = {
        "n_train": int(len(frame)),
        "n_test_sealed": len(test_ids),
        "use_description": description_has_text(frame),
        "class_counts": {str(label): int(count) for label, count in frame["y"].value_counts().items()},
    }
    return frame, info


def formulation_b_holdout_frame(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Bucket-labelled sealed-test rows. Unrated holdout ids are counted and excluded.

    Call this only from the one-shot evaluation. It loads the existing split file and
    does not draw a new one when that file is present.
    """
    population, test_ids = _prepare_population(config)
    holdout = population.loc[population["show_id"].astype(str).isin(test_ids)].copy()
    if len(holdout) != len(test_ids):
        missing = len(test_ids) - len(holdout)
        raise RuntimeError(f"Sealed split lists {missing} show_ids that are not in the catalogue.")
    n_unrated = int(holdout["true_ordinal"].isna().sum())
    frame = _labelled_frame(holdout, config)
    if set(frame["show_id"].astype(str)) - test_ids:
        raise RuntimeError("Holdout frame contains a show_id outside the sealed test list.")
    train_ids = set(population.loc[~population["show_id"].astype(str).isin(test_ids), "show_id"].astype(str))
    leaked = set(frame["show_id"].astype(str)).intersection(train_ids)
    if leaked:
        raise RuntimeError(f"Training show_ids entered the holdout frame: {len(leaked)}")
    info = {
        "n_holdout_ids": len(test_ids),
        "n_unrated_excluded": n_unrated,
        "n_scored": int(len(frame)),
        "class_counts": {str(label): int(count) for label, count in frame["y"].value_counts().items()},
    }
    return frame, info
