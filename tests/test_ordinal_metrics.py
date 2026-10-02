"""Hand-computed ordinal metrics, including the perfect and all-wrong edges."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from src.evaluation.ordinal import (
    adjacent_accuracy,
    kids_mature_confusions,
    ordinal_mae,
    quadratic_weighted_kappa,
    severe_error_rate,
)


def test_qwk_matches_the_hand_computed_two_class_case():
    # Confusion O = [[1, 1], [0, 2]]
    # Expected E = [[0.5, 1.5], [0.5, 1.5]]
    # Weights W = [[0, 1], [1, 0]] because (K - 1)^2 = 1
    # numerator = 1, denominator = 2, kappa = 0.5
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    score = quadratic_weighted_kappa(y_true, y_pred, labels=[0, 1])
    assert score == pytest.approx(0.5)
    assert score == pytest.approx(cohen_kappa_score(y_true, y_pred, labels=[0, 1], weights="quadratic"))


def test_qwk_perfect_predictions_are_one():
    y = [0, 1, 2, 3, 3, 0]
    assert quadratic_weighted_kappa(y, y, labels=[0, 1, 2, 3]) == pytest.approx(1.0)


def test_qwk_complete_swap_on_two_classes_is_minus_one():
    # O = [[0, 2], [2, 0]], E = [[1, 1], [1, 1]], num = 4, den = 2, kappa = -1
    y_true = [0, 0, 1, 1]
    y_pred = [1, 1, 0, 0]
    score = quadratic_weighted_kappa(y_true, y_pred, labels=[0, 1])
    assert score == pytest.approx(-1.0)
    assert score == pytest.approx(cohen_kappa_score(y_true, y_pred, labels=[0, 1], weights="quadratic"))


def test_ordinal_mae_adjacent_and_severe_match_hand_values():
    y_true = [0, 1, 2]
    y_pred = [0, 2, 2]
    # absolute errors 0, 1, 0 → mean 1/3; all within 1; none off by 2+
    assert ordinal_mae(y_true, y_pred) == pytest.approx(1 / 3)
    assert adjacent_accuracy(y_true, y_pred) == pytest.approx(1.0)
    assert severe_error_rate(y_true, y_pred) == pytest.approx(0.0)


def test_end_swap_counts_and_rates():
    y_true = [0, 3, 1, 3]
    y_pred = [3, 0, 1, 2]
    swaps = kids_mature_confusions(y_true, y_pred, kids_level=0, mature_level=3)
    assert swaps == {"kids_as_mature": 1, "mature_as_kids": 1}
    # errors of 3, 3, 0, 1 → two of four are severe
    assert severe_error_rate(y_true, y_pred) == pytest.approx(0.5)
    assert ordinal_mae(y_true, y_pred) == pytest.approx((3 + 3 + 0 + 1) / 4)


def test_qwk_on_the_full_ladder_matches_the_hand_computed_value():
    # y_true = [0, 0, 3, 3], y_pred = [0, 1, 3, 3], labels 0..3
    # O has one (0→1) error and three agreements.
    # numerator = (1/3)^2 = 1/9
    # denominator = 16/9
    # kappa = 1 - (1/9) / (16/9) = 15/16
    y_true = [0, 0, 3, 3]
    y_pred = [0, 1, 3, 3]
    score = quadratic_weighted_kappa(y_true, y_pred, labels=[0, 1, 2, 3])
    assert score == pytest.approx(15 / 16)
    assert score == pytest.approx(cohen_kappa_score(y_true, y_pred, labels=[0, 1, 2, 3], weights="quadratic"))
    # Dropping the unused level 2 renumbers the remaining labels and changes the weights.
    compacted = quadratic_weighted_kappa(y_true, y_pred, labels=[0, 1, 3])
    assert compacted != pytest.approx(score)


def test_metrics_reject_empty_input():
    with pytest.raises(ValueError):
        quadratic_weighted_kappa(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        ordinal_mae([], [])
    with pytest.raises(ValueError):
        adjacent_accuracy([], [])
    with pytest.raises(ValueError):
        severe_error_rate([], [])
