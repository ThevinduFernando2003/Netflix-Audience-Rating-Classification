"""Multiclass scores used beside the ordinal metrics."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.evaluation.ordinal import ordinal_bundle

ORDINAL_LABELS: list[int] = [0, 1, 2, 3]


def native_scores(y_true: Sequence[object], y_pred: Sequence[object]) -> dict[str, float]:
    """Accuracy, balanced accuracy, and macro-F1. Macro-F1 does not reward the majority class alone."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def score_mapped_ordinal(
    true_ordinal: np.ndarray,
    pred_labels: Sequence[object],
    pred_to_ordinal: dict[str, int],
) -> dict[str, float | int]:
    """Ordinal metrics on rows whose true label is on the ladder.

    Predictions that are not on the ladder (OTHER, NR, UR) are excluded from QWK
    and counted in `unmapped_prediction_rate`. They are not silently given a level.
    """
    truth = np.asarray(true_ordinal, dtype=float)
    guessed = np.asarray(
        [pred_to_ordinal.get(str(label), np.nan) for label in pred_labels],
        dtype=float,
    )
    eligible = ~np.isnan(truth)
    n_eligible = int(eligible.sum())
    if n_eligible == 0:
        raise ValueError("No rows with a maturity-bucket label were available to score.")
    unmapped = eligible & np.isnan(guessed)
    both = eligible & ~np.isnan(guessed)
    result: dict[str, float | int] = {
        "n_eligible": n_eligible,
        "n_mapped": int(both.sum()),
        "unmapped_prediction_rate": float(unmapped.sum() / n_eligible),
        "unmapped_prediction_count": int(unmapped.sum()),
    }
    if int(both.sum()) == 0:
        result.update(
            {
                "qwk": float("nan"),
                "ordinal_mae": float("nan"),
                "adjacent_accuracy": float("nan"),
                "severe_error_rate": float("nan"),
                "kids_as_mature": 0,
                "mature_as_kids": 0,
            }
        )
        return result
    bundle = ordinal_bundle(
        truth[both].astype(int),
        guessed[both].astype(int),
        labels=ORDINAL_LABELS,
        kids_level=0,
        mature_level=3,
    )
    result.update(bundle)
    return result
