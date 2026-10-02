"""Ordinal metrics. These are hand-tested; report numbers depend on them."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def quadratic_weighted_kappa(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    labels: Sequence[int] | None = None,
) -> float:
    """Cohen's quadratic weighted kappa on an ordinal scale.

    Weights are (i - j)^2 / (K - 1)^2 over the provided label order. Pass the full
    ladder (for this project, 0..3) so a one-step error keeps the same weight in
    every fold, including folds that happen to miss a class.
    """
    true = np.asarray(list(y_true), dtype=int)
    pred = np.asarray(list(y_pred), dtype=int)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if true.size == 0:
        raise ValueError("quadratic_weighted_kappa requires at least one row.")
    if labels is None:
        scale = list(range(int(max(true.max(), pred.max())) + 1))
    else:
        scale = [int(label) for label in labels]
    if len(scale) < 2:
        return 1.0
    index = {label: position for position, label in enumerate(scale)}
    unknown = sorted(set(true.tolist() + pred.tolist()) - set(scale))
    if unknown:
        raise ValueError(f"Labels outside the ordinal scale: {unknown}")
    width = len(scale)
    observed = np.zeros((width, width), dtype=float)
    for actual, guessed in zip(true, pred, strict=True):
        observed[index[int(actual)], index[int(guessed)]] += 1.0
    total = observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total
    positions = np.arange(width, dtype=float)
    weights = (positions[:, None] - positions[None, :]) ** 2 / (width - 1) ** 2
    numerator = float((weights * observed).sum())
    denominator = float((weights * expected).sum())
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else 0.0
    return 1.0 - numerator / denominator


def ordinal_mae(y_true: Sequence[int] | np.ndarray, y_pred: Sequence[int] | np.ndarray) -> float:
    """Mean |predicted level - true level|."""
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    if true.size == 0:
        raise ValueError("ordinal_mae requires at least one row.")
    return float(np.mean(np.abs(pred - true)))


def adjacent_accuracy(y_true: Sequence[int] | np.ndarray, y_pred: Sequence[int] | np.ndarray) -> float:
    """Fraction of predictions within one ordinal step of the truth, including exact matches."""
    true = np.asarray(list(y_true), dtype=int)
    pred = np.asarray(list(y_pred), dtype=int)
    if true.size == 0:
        raise ValueError("adjacent_accuracy requires at least one row.")
    return float(np.mean(np.abs(pred - true) <= 1))


def severe_error_rate(y_true: Sequence[int] | np.ndarray, y_pred: Sequence[int] | np.ndarray) -> float:
    """Fraction of predictions at least two ordinal steps from the truth."""
    true = np.asarray(list(y_true), dtype=int)
    pred = np.asarray(list(y_pred), dtype=int)
    if true.size == 0:
        raise ValueError("severe_error_rate requires at least one row.")
    return float(np.mean(np.abs(pred - true) >= 2))


def kids_mature_confusions(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    kids_level: int = 0,
    mature_level: int = 3,
) -> dict[str, int]:
    """Count the two ends of the ladder being swapped.

    `mature_as_kids` is the safety-critical failure: mature titles labelled as kids' content.
    `kids_as_mature` is the reverse error (over-blocking a children's title).
    """
    true = np.asarray(list(y_true), dtype=int)
    pred = np.asarray(list(y_pred), dtype=int)
    return {
        "kids_as_mature": int(np.sum((true == kids_level) & (pred == mature_level))),
        "mature_as_kids": int(np.sum((true == mature_level) & (pred == kids_level))),
    }


def ordinal_bundle(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    labels: Sequence[int] | None = None,
    kids_level: int = 0,
    mature_level: int = 3,
) -> dict[str, float | int]:
    """QWK, MAE, adjacent accuracy, severe-error rate, and the two end-swap counts."""
    swaps = kids_mature_confusions(y_true, y_pred, kids_level=kids_level, mature_level=mature_level)
    return {
        "qwk": quadratic_weighted_kappa(y_true, y_pred, labels=labels),
        "ordinal_mae": ordinal_mae(y_true, y_pred),
        "adjacent_accuracy": adjacent_accuracy(y_true, y_pred),
        "severe_error_rate": severe_error_rate(y_true, y_pred),
        "n": int(len(list(y_true))),
        **swaps,
    }
