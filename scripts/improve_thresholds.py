"""Class-probability multipliers on cached training out-of-fold scores.

The sealed test set is not loaded. A multiplier is adopted only when both
out-of-fold accuracy and out-of-fold macro-F1 beat the unscaled random forest.
The multiplier list was fixed before this script was run.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np

from src.config import REPO_ROOT
from src.evaluation.holdout import score_predictions
from src.logging_conf import configure_logging

logger = logging.getLogger(__name__)

# Fixed before looking at these out-of-fold results. Unlisted classes stay at 1.
PRE_REGISTERED_SCALES: list[dict[str, float]] = [
    {"Kids": 1.25},
    {"Kids": 1.5},
    {"Older Kids": 1.25},
    {"Older Kids": 1.5},
    {"Teens": 1.25},
    {"Mature": 0.85},
    {"Mature": 0.75},
    {"Kids": 1.5, "Older Kids": 1.25, "Mature": 0.85},
]


def _load(name: str) -> tuple[np.ndarray, list[str]]:
    stored = np.load(REPO_ROOT / "reports" / "improvement_runs" / f"{name}_oof.npz", allow_pickle=False)
    return stored["proba"], [str(item) for item in stored["classes"].tolist()]


def _scale(proba: np.ndarray, classes: list[str], factors: dict[str, float]) -> np.ndarray:
    scaled = proba.copy()
    for label, factor in factors.items():
        scaled[:, classes.index(label)] *= factor
    return scaled


def _row(name: str, factors: dict[str, float], proba: np.ndarray, classes: list[str], y: list[str]) -> dict[str, Any]:
    predicted = [classes[int(index)] for index in proba.argmax(axis=1)]
    scores = score_predictions(y, predicted)
    return {
        "base": name,
        "scales": factors,
        "accuracy": scores["accuracy"],
        "f1_macro": scores["f1_macro"],
        "qwk": scores["qwk"],
        "mature_as_kids": scores["mature_as_kids"],
    }


def run() -> None:
    configure_logging()
    decision = json.loads((REPO_ROOT / "reports" / "improvement_decision.json").read_text(encoding="utf-8"))
    y = _labels_from_training_cache(decision)
    forest, classes = _load("random_forest")
    logistic, logistic_classes = _load("logistic_regression")
    boosting, boosting_classes = _load("hist_gradient_boosting")
    if logistic_classes != classes or boosting_classes != classes:
        raise RuntimeError("Cached class order does not match across models.")
    if len(y) != len(forest):
        raise RuntimeError("Cached probabilities do not match the training label count.")
    mix = (logistic + 2.0 * forest + boosting) / 4.0
    baseline = _row("random_forest", {}, forest, classes, y)
    rows = [baseline]
    for factors in PRE_REGISTERED_SCALES:
        rows.append(_row("random_forest", factors, _scale(forest, classes, factors), classes, y))
        rows.append(_row("vote_1_2_1", factors, _scale(mix, classes, factors), classes, y))
    eligible = []
    for row in rows:
        if not row["scales"]:
            continue
        if row["accuracy"] > baseline["accuracy"] and row["f1_macro"] > baseline["f1_macro"]:
            eligible.append(row)
    adopted = bool(eligible)
    chosen = max(eligible, key=lambda row: (row["f1_macro"], row["accuracy"], row["qwk"])) if eligible else None
    payload = {
        "adopted": adopted,
        "rule": "Adopt a pre-registered class-probability scale only when out-of-fold accuracy and macro-F1 both beat the unscaled random forest. The holdout is not loaded.",
        "baseline": baseline,
        "chosen": chosen,
        "candidates": rows,
    }
    path = REPO_ROOT / "reports" / "threshold_decision.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for row in rows:
        logger.info(
            "%s %s accuracy %.4f macro-F1 %.4f",
            row["base"],
            row["scales"] or "unscaled",
            row["accuracy"],
            row["f1_macro"],
        )
    logger.info("Threshold adopted=%s", adopted)


def _labels_from_training_cache(decision: dict[str, Any]) -> list[str]:
    """Rebuild training labels in the same order the out-of-fold cache used.

    formulation_b_training_frame is deterministic for a saved split, and the
    cache was written from that frame. Reloading it does not read holdout labels.
    """
    from src.config import load_config
    from src.data.dataset import formulation_b_training_frame

    frame, info = formulation_b_training_frame(load_config())
    if int(info["n_train"]) != int(decision["n_train"]):
        raise RuntimeError("Training row count changed since the out-of-fold cache was written.")
    return frame["y"].astype(str).tolist()


if __name__ == "__main__":
    run()
