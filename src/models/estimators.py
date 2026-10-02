"""Estimator factory for the formulation comparison. Hyperparameters come from config.yaml."""

from __future__ import annotations

import inspect
from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_SEED

MODEL_NAMES: list[str] = [
    "dummy_most_frequent",
    "dummy_stratified",
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "complement_nb",
]

NONNEGATIVE_MODELS: set[str] = {"complement_nb"}


def make_estimator(name: str, config: dict[str, Any], seed: int = RANDOM_SEED) -> Any:
    """Build one untuned comparison estimator. This is not the production search."""
    params = config["comparison"]["estimators"]
    if name == "dummy_most_frequent":
        return DummyClassifier(strategy="most_frequent")
    if name == "dummy_stratified":
        return DummyClassifier(strategy="stratified", random_state=seed)
    if name == "logistic_regression":
        raw = dict(params["logistic_regression"])
        raw["random_state"] = seed
        if "multi_class" in inspect.signature(LogisticRegression.__init__).parameters:
            raw["multi_class"] = "multinomial"
        return LogisticRegression(**raw)
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=seed, **params["decision_tree"])
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params["random_forest"])
    if name == "extra_trees":
        return ExtraTreesClassifier(random_state=seed, n_jobs=-1, **params["extra_trees"])
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed, **params["hist_gradient_boosting"])
    if name == "complement_nb":
        return ComplementNB(**params["complement_nb"])
    raise KeyError(f"Unknown estimator {name}. Known: {MODEL_NAMES}")
