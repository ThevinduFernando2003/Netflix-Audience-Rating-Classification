"""The HTTP API rejects a blank title and reports whether the model file is present."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.serving.api import app
from src.serving.predict import model_path


def test_health_matches_the_model_file() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_present"] is model_path().exists()


def test_a_blank_title_is_rejected() -> None:
    response = TestClient(app).post("/predict", json={"title": "", "type": "Movie"})
    assert response.status_code == 422
