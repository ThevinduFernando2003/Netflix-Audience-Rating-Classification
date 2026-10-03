"""Choose a soft-vote mix on training folds only. The sealed test set is not read.

The three members are the Formulation B winners already stored in
reports/tuning_runs. Their out-of-fold probabilities are mixed with a short,
pre-registered list of weights. A mix is adopted only when both out-of-fold
accuracy and out-of-fold macro-F1 beat the random forest alone.
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
os.environ["PYTHONPATH"] = str(_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

import numpy as np
from sklearn.model_selection import cross_val_predict

from scripts.train import apply_params
from src.config import RANDOM_SEED, REPO_ROOT, load_config
from src.data.dataset import formulation_b_training_frame
from src.data.split import stratified_folds
from src.evaluation.holdout import score_predictions
from src.logging_conf import configure_logging
from src.models.pipelines import make_model_pipeline

logger = logging.getLogger(__name__)

# Order is logistic regression, random forest, hist gradient boosting.
# These eight mixes were fixed before the out-of-fold probabilities were computed.
PRE_REGISTERED_WEIGHTS: list[tuple[float, float, float]] = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 1),
    (1, 2, 1),
    (2, 1, 1),
    (1, 1, 2),
    (1, 2, 2),
]
MEMBER_NAMES = ["logistic_regression", "random_forest", "hist_gradient_boosting"]


def _load_params(name: str) -> dict[str, Any]:
    path = REPO_ROOT / "reports" / "tuning_runs" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))["best_params"]


def _oof_proba(name: str, frame: Any, y: Any, config: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    cache = REPO_ROOT / "reports" / "improvement_runs" / f"{name}_oof.npz"
    classes = sorted(set(y.astype(str)))
    if cache.exists():
        logger.info("Reusing %s", cache)
        stored = np.load(cache, allow_pickle=False)
        return stored["proba"], [str(item) for item in stored["classes"].tolist()]
    pipeline = make_model_pipeline(name, config, use_description=False, seed=RANDOM_SEED)
    apply_params(pipeline, _load_params(name))
    folds = stratified_folds(y, int(config["split"]["n_splits"]), RANDOM_SEED)
    logger.info("Out-of-fold probabilities for %s", name)
    proba = cross_val_predict(pipeline, frame, y, cv=folds, method="predict_proba", n_jobs=1)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, proba=np.asarray(proba, dtype=np.float64), classes=np.asarray(classes))
    return np.asarray(proba, dtype=np.float64), classes


def _mix(weights: tuple[float, float, float], members: list[np.ndarray]) -> np.ndarray:
    total = float(sum(weights))
    mixed = np.zeros_like(members[0])
    for weight, proba in zip(weights, members, strict=True):
        mixed += (weight / total) * proba
    return mixed


def run() -> None:
    configure_logging()
    config = load_config()
    frame, info = formulation_b_training_frame(config)
    y = frame["y"].astype(str)
    logger.info("Training rows for the vote: %s. The holdout is not loaded.", info["n_train"])
    members: list[np.ndarray] = []
    class_lists: list[list[str]] = []
    for name in MEMBER_NAMES:
        proba, classes = _oof_proba(name, frame, y, config)
        members.append(proba)
        class_lists.append(classes)
    if any(item != class_lists[0] for item in class_lists):
        raise RuntimeError(f"Class column order differs across models: {class_lists}")
    classes = class_lists[0]

    rows: list[dict[str, Any]] = []
    for weights in PRE_REGISTERED_WEIGHTS:
        mixed = _mix(weights, members)
        predicted = [classes[int(index)] for index in mixed.argmax(axis=1)]
        scores = score_predictions(y, predicted)
        rows.append(
            {
                "weights": {"logistic_regression": weights[0], "random_forest": weights[1], "hist_gradient_boosting": weights[2]},
                "accuracy": scores["accuracy"],
                "f1_macro": scores["f1_macro"],
                "balanced_accuracy": scores["balanced_accuracy"],
                "qwk": scores["qwk"],
                "ordinal_mae": scores["ordinal_mae"],
                "adjacent_accuracy": scores["adjacent_accuracy"],
                "severe_error_rate": scores["severe_error_rate"],
                "mature_as_kids": scores["mature_as_kids"],
                "kids_as_mature": scores["kids_as_mature"],
            }
        )
        logger.info(
            "weights %s accuracy %.4f macro-F1 %.4f qwk %.4f",
            weights,
            scores["accuracy"],
            scores["f1_macro"],
            scores["qwk"],
        )

    forest = next(row for row in rows if row["weights"]["random_forest"] == 1 and row["weights"]["logistic_regression"] == 0 and row["weights"]["hist_gradient_boosting"] == 0)
    eligible = [
        row
        for row in rows
        if row["accuracy"] > forest["accuracy"] and row["f1_macro"] > forest["f1_macro"]
    ]
    if eligible:
        chosen = max(eligible, key=lambda row: (row["f1_macro"], row["accuracy"], row["qwk"]))
        adopted = True
    else:
        chosen = max(rows, key=lambda row: (row["f1_macro"], row["accuracy"], row["qwk"]))
        adopted = False

    payload = {
        "adopted": adopted,
        "rule": "Adopt a pre-registered weight mix only when out-of-fold accuracy and out-of-fold macro-F1 both exceed the random forest alone. Otherwise keep the saved forest. The sealed test set is not an input.",
        "member_order": MEMBER_NAMES,
        "forest_oof": forest,
        "chosen": chosen,
        "candidates": rows,
        "n_train": info["n_train"],
        "seed": RANDOM_SEED,
    }
    out = REPO_ROOT / "reports" / "improvement_decision.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = _render(payload)
    report_path = REPO_ROOT / "reports" / "improvement_cv.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Wrote %s. Adopted=%s", report_path, adopted)


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _render(payload: dict[str, Any]) -> str:
    lines = [
        "# Training-fold vote — Formulation B",
        "",
        "The sealed test set was not loaded. Weights below were fixed before these probabilities were computed.",
        "",
        payload["rule"],
        "",
        f"Adopted: **{payload['adopted']}**.",
        "",
        "| logistic regression | random forest | hist gradient boosting | accuracy | macro-F1 | balanced accuracy | QWK | Mature→Kids |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["candidates"]:
        weights = row["weights"]
        lines.append(
            "| {lr:g} | {rf:g} | {hgb:g} | {acc} | {f1} | {bal} | {qwk} | {swap} |".format(
                lr=weights["logistic_regression"],
                rf=weights["random_forest"],
                hgb=weights["hist_gradient_boosting"],
                acc=_fmt(row["accuracy"]),
                f1=_fmt(row["f1_macro"]),
                bal=_fmt(row["balanced_accuracy"]),
                qwk=_fmt(row["qwk"]),
                swap=row["mature_as_kids"],
            )
        )
    chosen = payload["chosen"]
    forest = payload["forest_oof"]
    lines.extend(
        [
            "",
            f"Random forest alone, out of fold: accuracy {_fmt(forest['accuracy'])}, macro-F1 {_fmt(forest['f1_macro'])}.",
            (
                "Chosen mix: "
                f"logistic regression {chosen['weights']['logistic_regression']:g}, "
                f"random forest {chosen['weights']['random_forest']:g}, "
                f"hist gradient boosting {chosen['weights']['hist_gradient_boosting']:g}, "
                f"accuracy {_fmt(chosen['accuracy'])}, macro-F1 {_fmt(chosen['f1_macro'])}."
            ),
            "",
        ]
    )
    if payload["adopted"]:
        lines.append("The mix beat the forest on both out-of-fold accuracy and out-of-fold macro-F1. `scripts/score_improved.py` may score it on the holdout once.")
    else:
        lines.append("No pre-registered mix beat the forest on both accuracy and macro-F1. The saved forest stays the model. The holdout is not scored again for this search.")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    run()
