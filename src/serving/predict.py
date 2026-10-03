"""Turn one title into an ordered maturity prediction.

The fitted pipeline lives in models/best_model.joblib. This module does not fit
anything and does not read the sealed test set.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import pandas as pd

from src.config import REPO_ROOT, load_config
from src.data.clean import clean_catalogue
from src.features.engineer import engineer_features
from src.models.pipelines import BUCKET_ORDER

SEVERE_ERROR_WARNING = (
    "Severe-error warning: a Mature title predicted as Kids is the safety-critical failure. "
    "This title was predicted as Kids. Check it before treating it as children's content."
)


def model_path() -> Path:
    return REPO_ROOT / "models" / "best_model.joblib"


def load_saved_model(path: Path | None = None) -> Any:
    location = path if path is not None else model_path()
    if not location.exists():
        raise FileNotFoundError(
            f"{location} is missing. Train with scripts/train.py or mount the models directory."
        )
    return joblib.load(location)


def prediction_frame(payload: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    """Build the same engineered columns the saved pipeline was fit on."""
    title = str(payload.get("title") or "").strip()
    if not title:
        raise ValueError("title is required.")
    kind = str(payload.get("type") or "").strip()
    allowed = set(config["allowed_types"])
    if kind not in allowed:
        raise ValueError(f"type must be one of: {', '.join(sorted(allowed))}.")
    release_year = payload.get("release_year")
    row = pd.DataFrame(
        [
            {
                "show_id": "request",
                "type": kind,
                "title": title,
                "director": str(payload.get("director") or ""),
                "cast": str(payload.get("cast") or ""),
                "country": str(payload.get("country") or ""),
                "date_added": str(payload.get("date_added") or ""),
                "release_year": "" if release_year is None else str(release_year),
                "rating": "",
                "duration": str(payload.get("duration") or ""),
                "listed_in": str(payload.get("listed_in") or ""),
                "description": str(payload.get("description") or ""),
            }
        ]
    )
    cleaned = clean_catalogue(row, config)
    return engineer_features(cleaned, config)


def score_title(model: Any, config: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Return the four buckets in ordinal order, plus a warning when the call is Kids."""
    frame = prediction_frame(payload, config)
    predicted = str(model.predict(frame)[0])
    probabilities = model.predict_proba(frame)[0]
    classes = [str(label) for label in model.classes_]
    by_name = {label: float(probabilities[index]) for index, label in enumerate(classes)}
    missing = [name for name in BUCKET_ORDER if name not in by_name]
    if missing:
        raise RuntimeError(f"Model is missing bucket probabilities: {missing}")
    buckets = [
        {"name": name, "ordinal": index, "probability": by_name[name]}
        for index, name in enumerate(BUCKET_ORDER)
    ]
    warning = SEVERE_ERROR_WARNING if predicted == "Kids" else None
    return {
        "prediction": predicted,
        "ordinal": BUCKET_ORDER.index(predicted),
        "buckets": buckets,
        "severe_error_warning": warning,
    }
