"""Label formulations. The bucket map is read from config.yaml, not hardcoded."""

from __future__ import annotations

from typing import Any

import pandas as pd


class LabelMapError(ValueError):
    """A rating value is not in the config map, the unrated list, or the contamination rule."""


def bucket_table(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bucket in config["maturity_buckets"]:
        for rating in bucket["ratings"]:
            rows.append(
                {
                    "rating": str(rating),
                    "bucket": str(bucket["name"]),
                    "ordinal": int(bucket["ordinal"]),
                }
            )
    table = pd.DataFrame(rows)
    if table["rating"].duplicated().any():
        dupes = table.loc[table["rating"].duplicated(), "rating"].tolist()
        raise LabelMapError(f"Rating listed in more than one bucket: {dupes}")
    levels = sorted(int(bucket["ordinal"]) for bucket in config["maturity_buckets"])
    expected = list(range(len(levels)))
    if levels != expected:
        raise LabelMapError(f"Ordinal levels must be contiguous from 0. Got {levels}.")
    return table


def rating_to_bucket(config: dict[str, Any]) -> dict[str, tuple[str, int]]:
    table = bucket_table(config)
    return {str(row.rating): (str(row.bucket), int(row.ordinal)) for row in table.itertuples(index=False)}


def known_rating_tokens(config: dict[str, Any]) -> set[str]:
    mapped = set(rating_to_bucket(config))
    unrated = {str(token) for token in config["unrated_labels"]}
    overlap = mapped & unrated
    if overlap:
        raise LabelMapError(f"Unrated labels are also on the maturity ladder: {sorted(overlap)}")
    return mapped | unrated


def assert_ratings_covered(observed: list[object] | set[object], config: dict[str, Any]) -> None:
    """Fail if a rating value is neither mapped, unrated, blank, nor a duration leak."""
    from src.data.clean import is_contaminated_rating, is_missing_text

    known = known_rating_tokens(config)
    unknown: list[str] = []
    for value in observed:
        if is_missing_text(value) or is_contaminated_rating(value, config):
            continue
        token = str(value).strip()
        if token not in known:
            unknown.append(token)
    if unknown:
        raise LabelMapError(
            "Unmapped rating values: "
            + ", ".join(sorted(set(unknown)))
            + ". Add each one to maturity_buckets or unrated_labels in config.yaml."
        )


def assert_ordinal_levels_contiguous(config: dict[str, Any]) -> None:
    bucket_table(config)


def certification_system(rating: str, config: dict[str, Any]) -> str:
    if rating in set(config["film_ratings"]):
        return "film"
    if rating in set(config["tv_ratings"]):
        return "tv"
    if rating in set(config["unrated_labels"]):
        return "unrated"
    return "other"


def stabilize_raw_labels(
    ratings: pd.Series,
    *,
    min_count: int,
    n_splits: int,
    other_label: str,
) -> tuple[pd.Series, dict[str, object]]:
    """Fold raw classes smaller than min_count into OTHER, then drop OTHER if it cannot be stratified.

    Returns a series aligned to `ratings`. Dropped rows are pd.NA.
    """
    y = ratings.astype(str).copy()
    counts = y.value_counts()
    rare = sorted(label for label, count in counts.items() if int(count) < min_count)
    folded = {label: int(counts[label]) for label in rare}
    if rare:
        y = y.where(~y.isin(rare), other_label)
    report: dict[str, object] = {"folded_into_other": folded, "dropped_other": 0, "dropped_still_small": {}}
    counts_after = y.value_counts()
    if other_label in counts_after.index and int(counts_after[other_label]) < n_splits:
        report["dropped_other"] = int(counts_after[other_label])
        y = y.mask(y == other_label, pd.NA)
    remaining = y.dropna().value_counts()
    still_small = {label: int(count) for label, count in remaining.items() if int(count) < n_splits}
    if still_small:
        report["dropped_still_small"] = still_small
        y = y.mask(y.isin(list(still_small)), pd.NA)
    return y, report


def training_population(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Rows whose rating is a real catalogue value, including NR and UR."""
    return cleaned.loc[cleaned["include_in_training"]].copy()


def bucket_labels(ratings: pd.Series, config: dict[str, Any]) -> pd.DataFrame:
    """Map ratings to buckets. Unrated and unknown ratings get nulls — they are not given a level."""
    mapping = rating_to_bucket(config)
    buckets: list[object] = []
    ordinals: list[object] = []
    for rating in ratings.astype(str):
        found = mapping.get(rating)
        if found is None:
            buckets.append(pd.NA)
            ordinals.append(pd.NA)
        else:
            buckets.append(found[0])
            ordinals.append(found[1])
    return pd.DataFrame({"bucket": buckets, "ordinal": ordinals}, index=ratings.index)
