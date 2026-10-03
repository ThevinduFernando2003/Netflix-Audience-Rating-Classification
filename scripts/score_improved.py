"""Score the frozen soft vote once, and only if training folds already adopted it.

If reports/improvement_decision.json says the vote was not adopted, this script
does not load the holdout.
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

from sklearn.ensemble import VotingClassifier

from scripts.train import apply_params
from src.config import RANDOM_SEED, REPO_ROOT, load_config
from src.data.dataset import formulation_b_holdout_frame, formulation_b_training_frame
from src.evaluation.holdout import expected_calibration_error, score_predictions
from src.logging_conf import configure_logging
from src.models.pipelines import BUCKET_ORDER, make_model_pipeline
from src.models.persistence import save_model, utc_now

logger = logging.getLogger(__name__)


def _member(name: str, config: dict[str, Any]) -> Any:
    params = json.loads((REPO_ROOT / "reports" / "tuning_runs" / f"{name}.json").read_text(encoding="utf-8"))
    pipeline = make_model_pipeline(name, config, use_description=False, seed=RANDOM_SEED)
    return apply_params(pipeline, params["best_params"])


def run() -> None:
    configure_logging()
    decision_path = REPO_ROOT / "reports" / "improvement_decision.json"
    if not decision_path.exists():
        raise SystemExit("reports/improvement_decision.json is missing. Run scripts/improve_cv.py first.")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if not decision["adopted"]:
        logger.info("Vote was not adopted on training folds. Holdout stays untouched.")
        note = REPO_ROOT / "reports" / "improved_test_evaluation.md"
        note.write_text(
            "# Improved model — not scored\n\n"
            "The pre-registered vote did not beat the random forest on both out-of-fold accuracy and out-of-fold macro-F1. "
            "The holdout was not loaded for this step. The sealed score remains `reports/test_evaluation.md`.\n",
            encoding="utf-8",
        )
        return

    config = load_config()
    weights = decision["chosen"]["weights"]
    ordered = [float(weights[name]) for name in decision["member_order"]]
    vote = VotingClassifier(
        estimators=[(name, _member(name, config)) for name in decision["member_order"]],
        voting="soft",
        weights=ordered,
        n_jobs=1,
    )
    train, train_info = formulation_b_training_frame(config)
    holdout, holdout_info = formulation_b_holdout_frame(config)
    y_train = train["y"].astype(str)
    logger.info("Fitting the frozen vote on %s training rows", train_info["n_train"])
    vote.fit(train, y_train)
    pred = vote.predict(holdout)
    proba = vote.predict_proba(holdout)
    scores = score_predictions(holdout["y"].astype(str), pred)
    calibration = expected_calibration_error(holdout["y"].astype(str), proba, [str(label) for label in vote.classes_])
    model_path = REPO_ROOT / "models" / "improved_model.joblib"
    metadata_path = REPO_ROOT / "models" / "improved_metadata.json"
    save_model(
        vote,
        model_path,
        {
            "model_name": "soft_vote",
            "formulation": "B",
            "label_names_in_ordinal_order": BUCKET_ORDER,
            "trained_at": utc_now(),
            "weights": weights,
            "n_train": train_info["n_train"],
            "n_test_sealed_unscored_before_this_script": holdout_info["n_holdout_ids"],
            "random_seed": RANDOM_SEED,
            "oof_accuracy": decision["chosen"]["accuracy"],
            "oof_f1_macro": decision["chosen"]["f1_macro"],
            "test_set_scored": True,
            "selection_used_test_set": False,
        },
        metadata_path,
    )
    payload = {
        "n_scored": holdout_info["n_scored"],
        "n_unrated_excluded": holdout_info["n_unrated_excluded"],
        "weights": weights,
        "holdout": scores,
        "calibration_ece": calibration["ece"],
    }
    (REPO_ROOT / "reports" / "improved_test_evaluation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Sealed score of the frozen soft vote",
        "",
        "The weight mix was chosen in `reports/improvement_cv.md` before this file was written. These holdout numbers were not used to pick the mix.",
        "",
        f"- Rows scored: **{holdout_info['n_scored']}**. Unrated ids excluded: **{holdout_info['n_unrated_excluded']}**.",
        f"- Weights: logistic regression {weights['logistic_regression']:g}, random forest {weights['random_forest']:g}, hist gradient boosting {weights['hist_gradient_boosting']:g}.",
        "",
        "| metric | holdout |",
        "|---|---:|",
        f"| accuracy | {scores['accuracy']:.4f} |",
        f"| macro-F1 | {scores['f1_macro']:.4f} |",
        f"| balanced accuracy | {scores['balanced_accuracy']:.4f} |",
        f"| quadratic weighted kappa | {scores['qwk']:.4f} |",
        f"| ordinal MAE | {scores['ordinal_mae']:.4f} |",
        f"| adjacent accuracy | {scores['adjacent_accuracy']:.4f} |",
        f"| severe-error rate | {scores['severe_error_rate']:.4f} |",
        f"| Mature predicted as Kids | {scores['mature_as_kids']} |",
        f"| Kids predicted as Mature | {scores['kids_as_mature']} |",
        f"| top-label ECE | {calibration['ece']:.4f} |",
        "",
        "The original forest score remains in `reports/test_evaluation.md`. Compare the two files. Do not retune from either.",
        "",
    ]
    (REPO_ROOT / "reports" / "improved_test_evaluation.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote reports/improved_test_evaluation.md")


if __name__ == "__main__":
    run()