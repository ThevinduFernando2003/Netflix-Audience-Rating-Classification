"""Metrics for the one-shot sealed-test report. No fitting happens here."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.evaluation.ordinal import kids_mature_confusions, ordinal_bundle
from src.evaluation.scoring import native_scores
from src.models.pipelines import BUCKET_ORDER

ORDINAL = {name: index for index, name in enumerate(BUCKET_ORDER)}


def expected_calibration_error(
    y_true: Sequence[str],
    proba: np.ndarray,
    classes: Sequence[str],
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Top-label expected calibration error. Confidence is the maximum class probability."""
    class_index = {str(label): index for index, label in enumerate(classes)}
    truth = [class_index[str(label)] for label in y_true]
    confidence = proba.max(axis=1)
    predicted = proba.argmax(axis=1)
    correct = (predicted == np.asarray(truth, dtype=int)).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float | int]] = []
    ece = 0.0
    for bin_id in range(n_bins):
        low, high = float(edges[bin_id]), float(edges[bin_id + 1])
        if bin_id == 0:
            mask = (confidence >= low) & (confidence <= high)
        else:
            mask = (confidence > low) & (confidence <= high)
        count = int(mask.sum())
        if count == 0:
            continue
        accuracy = float(correct[mask].mean())
        mean_confidence = float(confidence[mask].mean())
        weight = count / len(confidence)
        ece += weight * abs(accuracy - mean_confidence)
        rows.append(
            {
                "bin_low": low,
                "bin_high": high,
                "count": count,
                "accuracy": accuracy,
                "confidence": mean_confidence,
            }
        )
    return {"ece": float(ece), "bins": rows}


def score_predictions(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, Any]:
    """Native scores, ordinal scores, per-class F1, and the confusion matrix."""
    truth = [str(label) for label in y_true]
    guess = [str(label) for label in y_pred]
    true_ord = [ORDINAL[label] for label in truth]
    pred_ord = [ORDINAL[label] for label in guess]
    native = native_scores(truth, guess)
    ordinal = ordinal_bundle(true_ord, pred_ord, labels=[0, 1, 2, 3])
    swaps = kids_mature_confusions(true_ord, pred_ord)
    report = classification_report(truth, guess, labels=BUCKET_ORDER, output_dict=True, zero_division=0)
    matrix = confusion_matrix(truth, guess, labels=BUCKET_ORDER)
    per_class = {
        label: {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        for label in BUCKET_ORDER
    }
    return {
        "n": len(truth),
        "accuracy": native["accuracy"],
        "balanced_accuracy": native["balanced_accuracy"],
        "f1_macro": native["f1_macro"],
        "qwk": float(ordinal["qwk"]),
        "ordinal_mae": float(ordinal["ordinal_mae"]),
        "adjacent_accuracy": float(ordinal["adjacent_accuracy"]),
        "severe_error_rate": float(ordinal["severe_error_rate"]),
        "mature_as_kids": int(swaps["mature_as_kids"]),
        "kids_as_mature": int(swaps["kids_as_mature"]),
        "per_class": per_class,
        "confusion_matrix": matrix.astype(int).tolist(),
    }
