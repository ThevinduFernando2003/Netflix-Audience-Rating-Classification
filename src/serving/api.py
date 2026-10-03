"""HTTP API for one Formulation B prediction. The model file is loaded at startup."""

from __future__ import annotations

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import load_config
from src.serving.predict import load_saved_model, model_path, score_title

app = FastAPI(title="Netflix audience rating", version="1.0.0")
_model = None
_config = None


class TitleRequest(BaseModel):
    title: str = Field(min_length=1)
    type: str
    listed_in: str = ""
    director: str = ""
    country: str = ""
    date_added: str = ""
    release_year: int | None = None
    duration: str = ""
    cast: str = ""
    description: str = ""


def _get_model():
    global _model, _config
    if _model is None:
        _config = load_config()
        _model = load_saved_model()
    return _model, _config


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model_present": model_path().exists()}


@app.post("/predict")
def predict(body: TitleRequest) -> dict[str, object]:
    if not model_path().exists():
        raise HTTPException(status_code=503, detail="models/best_model.joblib is not available.")
    model, config = _get_model()
    try:
        return score_title(model, config, body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
