"""Prediction responses stay on the maturity ladder and flag a Kids call."""

from __future__ import annotations

import numpy as np

from src.config import load_config
from src.models.pipelines import BUCKET_ORDER
from src.serving.predict import SEVERE_ERROR_WARNING, prediction_frame, score_title


class _FakeModel:
    """classes_ is deliberately not in ordinal order."""

    classes_ = np.array(["Mature", "Kids", "Teens", "Older Kids"])

    def predict(self, frame):
        title = str(frame["title"].iloc[0]).lower()
        if "pony" in title:
            return np.array(["Kids"])
        return np.array(["Teens"])

    def predict_proba(self, frame):
        if self.predict(frame)[0] == "Kids":
            return np.array([[0.05, 0.80, 0.05, 0.10]])
        return np.array([[0.10, 0.05, 0.70, 0.15]])


def test_buckets_are_returned_in_ordinal_order_and_kids_is_warned():
    config = load_config()
    payload = {"title": "Pony Tale", "type": "Movie", "listed_in": "Children & Family Movies", "duration": "90 min"}
    result = score_title(_FakeModel(), config, payload)
    assert [item["name"] for item in result["buckets"]] == BUCKET_ORDER
    assert [item["ordinal"] for item in result["buckets"]] == [0, 1, 2, 3]
    assert result["prediction"] == "Kids"
    assert result["ordinal"] == 0
    assert result["severe_error_warning"] == SEVERE_ERROR_WARNING
    kids = result["buckets"][0]["probability"]
    assert kids == 0.80


def test_a_non_kids_prediction_has_no_severe_error_warning():
    config = load_config()
    result = score_title(_FakeModel(), config, {"title": "City Lights", "type": "Movie", "duration": "100 min"})
    assert result["prediction"] == "Teens"
    assert result["severe_error_warning"] is None
    assert result["buckets"][2]["probability"] == 0.70


def test_prediction_frame_keeps_an_empty_description():
    config = load_config()
    frame = prediction_frame({"title": "City Lights", "type": "TV Show", "duration": "2 Seasons"}, config)
    assert frame["description"].iloc[0] == ""
    assert frame["title"].iloc[0] == "City Lights"
