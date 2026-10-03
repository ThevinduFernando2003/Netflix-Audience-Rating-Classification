"""Score the sealed Formulation B test set once.

The saved random forest is the model chosen by training macro-F1. This script
does not change its parameters. A class_weight=None twin is fit on the training
pool and scored on the same holdout as a pre-specified comparison. That
comparison is reported. It does not replace the saved model.
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

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone

from src.config import REPO_ROOT, load_config
from src.data.dataset import formulation_b_holdout_frame, formulation_b_training_frame
from src.evaluation.holdout import expected_calibration_error, score_predictions
from src.logging_conf import configure_logging
from src.models.pipelines import BUCKET_ORDER

logger = logging.getLogger(__name__)


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _section(title: str, scores: dict[str, Any], calibration: dict[str, Any], train_scores: dict[str, Any]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| metric | holdout | training rows, same fitted model |",
        "|---|---:|---:|",
        f"| accuracy | {_fmt(scores['accuracy'])} | {_fmt(train_scores['accuracy'])} |",
        f"| macro-F1 | {_fmt(scores['f1_macro'])} | {_fmt(train_scores['f1_macro'])} |",
        f"| balanced accuracy | {_fmt(scores['balanced_accuracy'])} | {_fmt(train_scores['balanced_accuracy'])} |",
        f"| quadratic weighted kappa | {_fmt(scores['qwk'])} | {_fmt(train_scores['qwk'])} |",
        f"| ordinal MAE | {_fmt(scores['ordinal_mae'])} | {_fmt(train_scores['ordinal_mae'])} |",
        f"| adjacent accuracy | {_fmt(scores['adjacent_accuracy'])} | {_fmt(train_scores['adjacent_accuracy'])} |",
        f"| severe-error rate | {_fmt(scores['severe_error_rate'])} | {_fmt(train_scores['severe_error_rate'])} |",
        f"| Mature predicted as Kids | {scores['mature_as_kids']} | {train_scores['mature_as_kids']} |",
        f"| Kids predicted as Mature | {scores['kids_as_mature']} | {train_scores['kids_as_mature']} |",
        "",
        f"Holdout macro-F1 minus training macro-F1: **{_fmt(scores['f1_macro'] - train_scores['f1_macro'])}**.",
        f"Top-label expected calibration error on the holdout: **{_fmt(calibration['ece'])}**.",
        "",
        "| class | precision | recall | F1 | support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in BUCKET_ORDER:
        row = scores["per_class"][label]
        lines.append(
            f"| {label} | {_fmt(row['precision'])} | {_fmt(row['recall'])} | {_fmt(row['f1'])} | {row['support']} |"
        )
    lines.extend(["", "Confusion matrix, rows are the true bucket and columns are the prediction.", ""])
    header = "| true \\ predicted | " + " | ".join(BUCKET_ORDER) + " |"
    lines.append(header)
    lines.append("|---|" + "---:|" * len(BUCKET_ORDER))
    for label, row in zip(BUCKET_ORDER, scores["confusion_matrix"], strict=True):
        lines.append("| " + label + " | " + " | ".join(str(int(item)) for item in row) + " |")
    lines.extend(["", "| confidence bin | count | accuracy in bin | mean confidence |", "|---|---:|---:|---:|"])
    for bin_row in calibration["bins"]:
        lines.append(
            f"| {_fmt(bin_row['bin_low'], 2)}–{_fmt(bin_row['bin_high'], 2)} | {bin_row['count']} | {_fmt(bin_row['accuracy'])} | {_fmt(bin_row['confidence'])} |"
        )
    lines.append("")
    return lines


def _plot_confusion(scores: dict[str, Any], path: Path, title: str) -> None:
    matrix = np.asarray(scores["confusion_matrix"], dtype=float)
    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(BUCKET_ORDER)), BUCKET_ORDER, rotation=30, ha="right")
    axis.set_yticks(range(len(BUCKET_ORDER)), BUCKET_ORDER)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)
    for row_id, row in enumerate(matrix):
        for col_id, value in enumerate(row):
            axis.text(col_id, row_id, str(int(value)), ha="center", va="center", color="black")
    fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_calibration(curves: list[tuple[str, dict[str, Any]]], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="perfect")
    for name, calibration in curves:
        if not calibration["bins"]:
            continue
        confidence = [row["confidence"] for row in calibration["bins"]]
        accuracy = [row["accuracy"] for row in calibration["bins"]]
        axis.plot(confidence, accuracy, marker="o", label=name)
    axis.set_xlabel("Mean confidence in bin")
    axis.set_ylabel("Accuracy in bin")
    axis.set_title("Holdout reliability")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _predict_aligned(model: Any, frame: Any) -> tuple[np.ndarray, np.ndarray, list[str]]:
    predicted = model.predict(frame)
    proba = model.predict_proba(frame)
    classes = [str(label) for label in model.classes_]
    return np.asarray(predicted), np.asarray(proba), classes


def run() -> None:
    configure_logging()
    config = load_config()
    model_path = REPO_ROOT / "models" / "best_model.joblib"
    if not model_path.exists():
        raise SystemExit(f"Missing {model_path}. Run scripts/train.py first.")
    saved = joblib.load(model_path)
    train, train_info = formulation_b_training_frame(config)
    holdout, holdout_info = formulation_b_holdout_frame(config)
    overlap = set(train["show_id"].astype(str)).intersection(set(holdout["show_id"].astype(str)))
    if overlap:
        raise RuntimeError(f"Train and holdout share {len(overlap)} show_ids.")
    logger.info(
        "Scoring %s labelled holdout rows (%s unrated ids excluded from %s sealed ids)",
        holdout_info["n_scored"],
        holdout_info["n_unrated_excluded"],
        holdout_info["n_holdout_ids"],
    )

    y_test = holdout["y"].astype(str)
    y_train = train["y"].astype(str)
    pred_test, proba_test, classes = _predict_aligned(saved, holdout)
    pred_train, _, _ = _predict_aligned(saved, train)
    balanced_test = score_predictions(y_test, pred_test)
    balanced_train = score_predictions(y_train, pred_train)
    balanced_calibration = expected_calibration_error(y_test, proba_test, classes)

    logger.info("Fitting the class_weight=None twin on training rows only")
    twin = clone(saved)
    twin.set_params(clf__class_weight=None)
    twin.fit(train, y_train)
    twin_pred, twin_proba, twin_classes = _predict_aligned(twin, holdout)
    twin_train_pred, _, _ = _predict_aligned(twin, train)
    unweighted_test = score_predictions(y_test, twin_pred)
    unweighted_train = score_predictions(y_train, twin_train_pred)
    unweighted_calibration = expected_calibration_error(y_test, twin_proba, twin_classes)

    figure_dir = REPO_ROOT / "reports" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    _plot_confusion(
        balanced_test,
        figure_dir / "confusion_matrix_test.png",
        "Holdout confusion, balanced_subsample forest",
    )
    _plot_confusion(
        unweighted_test,
        figure_dir / "confusion_matrix_test_unweighted.png",
        "Holdout confusion, unweighted forest",
    )
    _plot_calibration(
        [
            ("balanced_subsample", balanced_calibration),
            ("class_weight=None", unweighted_calibration),
        ],
        figure_dir / "calibration_test.png",
    )

    cv_path = REPO_ROOT / "reports" / "best_params.json"
    cv_scores = json.loads(cv_path.read_text(encoding="utf-8")).get("cv_scores", {}) if cv_path.exists() else {}
    payload = {
        "formulation": "B",
        "model_path": str(model_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "parameters_changed": False,
        "n_train": train_info["n_train"],
        "n_holdout_ids": holdout_info["n_holdout_ids"],
        "n_unrated_excluded": holdout_info["n_unrated_excluded"],
        "n_scored": holdout_info["n_scored"],
        "holdout_class_counts": holdout_info["class_counts"],
        "cv_scores_from_tuning": cv_scores,
        "balanced_subsample": {
            "holdout": balanced_test,
            "train_resubstitution": balanced_train,
            "calibration": balanced_calibration,
        },
        "class_weight_none": {
            "holdout": unweighted_test,
            "train_resubstitution": unweighted_train,
            "calibration": unweighted_calibration,
        },
    }
    json_path = REPO_ROOT / "reports" / "test_evaluation.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    cv_f1 = cv_scores.get("f1_macro", {})
    lines = [
        "# Sealed test evaluation — Formulation B",
        "",
        "This is the one-shot score of the random forest chosen in `reports/tuning_results.md`.",
        "No parameter was changed from these numbers. The unweighted forest is a pre-specified comparison, not a new search, and it does not replace the saved model.",
        "",
        "## What was scored",
        "",
        f"- Sealed ids: **{holdout_info['n_holdout_ids']}**.",
        f"- Unrated ids (NR/UR) excluded, because they are not on the ladder: **{holdout_info['n_unrated_excluded']}**.",
        f"- Rows scored: **{holdout_info['n_scored']}**.",
        f"- Training rows the models were fit on: **{train_info['n_train']}**.",
        "- The training column below is resubstitution: the fitted model predicting the rows it was trained on. It is the memorisation check, not a second validation fold.",
        "",
        "During tuning, the same forest's 5-fold training macro-F1 was "
        f"{_fmt(float(cv_f1.get('train_mean', float('nan'))))} ± {_fmt(float(cv_f1.get('train_std', float('nan'))))} "
        f"and its validation macro-F1 was {_fmt(float(cv_f1.get('test_mean', float('nan'))))} ± {_fmt(float(cv_f1.get('test_std', float('nan'))))}.",
        "",
    ]
    lines.extend(
        _section(
            "Saved model: random forest, class_weight=balanced_subsample",
            balanced_test,
            balanced_calibration,
            balanced_train,
        )
    )
    lines.append("Figure: `reports/figures/confusion_matrix_test.png`.")
    lines.append("")
    lines.extend(
        _section(
            "Pre-specified comparison: same forest, class_weight=None",
            unweighted_test,
            unweighted_calibration,
            unweighted_train,
        )
    )
    lines.extend(
        [
            "Figure: `reports/figures/confusion_matrix_test_unweighted.png`.",
            "",
            "Both reliability curves are in `reports/figures/calibration_test.png`.",
            "",
            "## How to read the gap and the class weights",
            "",
            (
                f"The saved forest's holdout macro-F1 is {_fmt(balanced_test['f1_macro'])} "
                f"against a training-row macro-F1 of {_fmt(balanced_train['f1_macro'])}. "
                f"Holdout accuracy is {_fmt(balanced_test['accuracy'])} against training-row accuracy {_fmt(balanced_train['accuracy'])}."
            ),
            "",
            (
                f"Removing class weights moves holdout accuracy from {_fmt(balanced_test['accuracy'])} to {_fmt(unweighted_test['accuracy'])} "
                f"and holdout macro-F1 from {_fmt(balanced_test['f1_macro'])} to {_fmt(unweighted_test['f1_macro'])}. "
                f"Mature predicted as Kids moves from {balanced_test['mature_as_kids']} to {unweighted_test['mature_as_kids']}."
            ),
            "",
            "The saved model stays the balanced forest. This table is the measurement, not a reason to retune.",
            "",
        ]
    )
    report_path = REPO_ROOT / "reports" / "test_evaluation.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    run()
