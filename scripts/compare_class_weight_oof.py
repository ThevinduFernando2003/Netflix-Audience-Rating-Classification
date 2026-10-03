"""Out-of-fold class_weight=None versus the saved balanced forest.

Uses the training pool only. Adopt nothing from the holdout. A no-weight
forest is preferred on this evidence only when both out-of-fold accuracy and
out-of-fold macro-F1 are higher than the balanced forest already cached.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

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


def run() -> None:
    configure_logging()
    config = load_config()
    frame, info = formulation_b_training_frame(config)
    y = frame["y"].astype(str)
    cache = REPO_ROOT / "reports" / "improvement_runs" / "random_forest_unweighted_oof.npz"
    classes = sorted(set(y.astype(str)))
    if cache.exists():
        logger.info("Reusing %s", cache)
        stored = np.load(cache, allow_pickle=False)
        proba = stored["proba"]
        classes = [str(item) for item in stored["classes"].tolist()]
    else:
        params = json.loads(
            (REPO_ROOT / "reports" / "tuning_runs" / "random_forest.json").read_text(encoding="utf-8")
        )["best_params"]
        pipeline = make_model_pipeline("random_forest", config, use_description=False, seed=RANDOM_SEED)
        apply_params(pipeline, params)
        pipeline.set_params(clf__class_weight=None)
        folds = stratified_folds(y, int(config["split"]["n_splits"]), RANDOM_SEED)
        logger.info("Out-of-fold probabilities for the unweighted forest")
        proba = cross_val_predict(pipeline, frame, y, cv=folds, method="predict_proba", n_jobs=1)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, proba=np.asarray(proba, dtype=np.float64), classes=np.asarray(classes))
    predicted = [classes[int(index)] for index in np.asarray(proba).argmax(axis=1)]
    scores = score_predictions(y, predicted)
    balanced = json.loads((REPO_ROOT / "reports" / "improvement_decision.json").read_text(encoding="utf-8"))
    forest = balanced["forest_oof"]
    better = scores["accuracy"] > forest["accuracy"] and scores["f1_macro"] > forest["f1_macro"]
    payload = {
        "adopted": better,
        "rule": "Prefer class_weight=None only when both out-of-fold accuracy and out-of-fold macro-F1 beat the balanced forest. The holdout table is not an input to this flag.",
        "n_train": info["n_train"],
        "unweighted_oof": {
            "accuracy": scores["accuracy"],
            "f1_macro": scores["f1_macro"],
            "balanced_accuracy": scores["balanced_accuracy"],
            "qwk": scores["qwk"],
            "mature_as_kids": scores["mature_as_kids"],
            "kids_as_mature": scores["kids_as_mature"],
        },
        "balanced_oof": {
            "accuracy": forest["accuracy"],
            "f1_macro": forest["f1_macro"],
            "qwk": forest["qwk"],
            "mature_as_kids": forest["mature_as_kids"],
        },
    }
    out = REPO_ROOT / "reports" / "class_weight_oof.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "Unweighted OOF accuracy %.4f macro-F1 %.4f. Balanced accuracy %.4f macro-F1 %.4f. Prefer unweighted=%s",
        scores["accuracy"],
        scores["f1_macro"],
        forest["accuracy"],
        forest["f1_macro"],
        better,
    )


if __name__ == "__main__":
    run()
