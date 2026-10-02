"""Formulation B model pipelines and the halving-search spaces.

Inner estimators use n_jobs=1. The search itself is the thing that runs in parallel.
"""

from __future__ import annotations

import inspect
from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_SEED
from src.features.build_pipeline import ToDense, make_preprocessor

BUCKET_ORDER: list[str] = ["Kids", "Older Kids", "Teens", "Mature"]


def make_model_pipeline(
    name: str,
    config: dict[str, Any],
    *,
    use_description: bool,
    seed: int = RANDOM_SEED,
) -> Pipeline:
    """Preprocess inside the pipeline so every CV fold fits its own vocabulary."""
    steps: list[tuple[str, Any]] = [
        ("preprocess", make_preprocessor(config, use_description=use_description, nonnegative=False)),
    ]
    if name == "hist_gradient_boosting":
        steps.append(("to_dense", ToDense()))
    steps.append(("clf", _estimator(name, seed)))
    return Pipeline(steps)


def _estimator(name: str, seed: int) -> Any:
    if name == "logistic_regression":
        raw: dict[str, Any] = {
            "C": 1.0,
            "max_iter": 400,
            "solver": "saga",
            "penalty": "l2",
            "class_weight": "balanced",
            "tol": 0.001,
            "random_state": seed,
            "n_jobs": 1,
        }
        if "multi_class" in inspect.signature(LogisticRegression.__init__).parameters:
            raw["multi_class"] = "multinomial"
        return LogisticRegression(**raw)
    if name == "decision_tree":
        return DecisionTreeClassifier(
            random_state=seed,
            class_weight="balanced",
            max_depth=12,
            min_samples_leaf=5,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            random_state=seed,
            n_jobs=1,
            class_weight="balanced_subsample",
            n_estimators=80,
            max_depth=16,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            random_state=seed,
            class_weight="balanced",
            learning_rate=0.1,
            max_iter=50,
            max_bins=64,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            early_stopping=False,
        )
    raise KeyError(f"No Formulation B search pipeline for {name}")


def title_vectorizer_space() -> dict[str, list[Any]]:
    return {
        "preprocess__title__tfidf__min_df": [2, 3, 5],
        "preprocess__title__tfidf__max_features": [2000, 3500, 5000],
        "preprocess__title__tfidf__ngram_range": [(1, 1), (1, 2)],
    }


def search_spaces() -> dict[str, dict[str, list[Any]]]:
    """Parameter lists for HalvingRandomSearchCV. Class weights stay balanced."""
    text = title_vectorizer_space()
    return {
        "logistic_regression": {
            **text,
            "clf__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        },
        "decision_tree": {
            **text,
            "clf__max_depth": [6, 10, 14, 20, None],
            "clf__min_samples_split": [2, 10, 20, 40],
            "clf__min_samples_leaf": [1, 5, 10, 20],
            "clf__max_features": ["sqrt", "log2", 0.3],
            "clf__criterion": ["gini", "entropy"],
            "clf__ccp_alpha": [0.0, 0.0005, 0.001, 0.005],
        },
        "random_forest": {
            **text,
            "clf__n_estimators": [60, 120, 200],
            "clf__max_depth": [12, 18, None],
            "clf__min_samples_split": [2, 8, 20],
            "clf__min_samples_leaf": [1, 2, 5],
            "clf__max_features": ["sqrt", 0.3],
            "clf__criterion": ["gini", "entropy"],
        },
        "hist_gradient_boosting": {
            **text,
            "clf__learning_rate": [0.05, 0.1, 0.15],
            "clf__max_iter": [30, 50],
            "clf__max_leaf_nodes": [15, 31, 63],
            "clf__min_samples_leaf": [10, 20, 40],
            "clf__l2_regularization": [0.0, 0.1, 1.0],
            "clf__max_bins": [32, 64],
        },
    }
