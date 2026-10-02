"""Sealed stratified holdout. The test ids are written once and not scored during label design."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from src.config import RANDOM_SEED


def stratification_key(bucket: object) -> str:
    if bucket is None or (isinstance(bucket, float) and pd.isna(bucket)) or pd.isna(bucket):
        return "UNRATED"
    text = str(bucket)
    if text in {"", "<NA>", "nan", "None"}:
        return "UNRATED"
    return text


def sealed_split(
    show_ids: pd.Series,
    strata: pd.Series,
    *,
    test_size: float,
    seed: int = RANDOM_SEED,
) -> dict[str, list[str]]:
    """Hold out `test_size` of rows, stratified. Returns show_id lists."""
    if len(show_ids) != len(strata):
        raise ValueError("show_ids and strata must be the same length.")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_positions, test_positions = next(splitter.split(show_ids.to_frame(), strata.astype(str)))
    ids = show_ids.astype(str).to_numpy()
    return {
        "train_show_ids": ids[train_positions].tolist(),
        "test_show_ids": ids[test_positions].tolist(),
    }


def save_sealed_split(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_sealed_split(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stratified_folds(y: pd.Series, n_splits: int, seed: int = RANDOM_SEED) -> StratifiedKFold:
    counts = y.astype(str).value_counts()
    too_small = {label: int(count) for label, count in counts.items() if int(count) < n_splits}
    if too_small:
        raise ValueError(
            f"StratifiedKFold(n_splits={n_splits}) cannot run. Classes below {n_splits} rows: {too_small}. "
            "Fix the label design rather than dropping stratification."
        )
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
