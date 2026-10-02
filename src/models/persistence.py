"""Write the fitted pipeline and the metadata that says the test set was not scored."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def save_model(pipeline: Any, model_path: Path, metadata: dict[str, Any], metadata_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
