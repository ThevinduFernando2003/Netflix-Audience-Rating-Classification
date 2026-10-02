"""Fold-fitted pipeline behaviour on the fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED, load_config
from src.data.clean import clean_catalogue
from src.data.labels import bucket_labels, training_population
from src.data.load import load_raw_catalogue
from src.data.split import stratified_folds
from src.features.build_pipeline import make_preprocessor
from src.features.engineer import description_has_text, engineer_features
from src.models.estimators import make_estimator


def _prepared():
    config = load_config()
    raw = load_raw_catalogue(Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mini_netflix.csv", config)
    cleaned = clean_catalogue(raw, config)
    trainable = training_population(cleaned)
    features = engineer_features(trainable, config)
    labels = bucket_labels(trainable["rating"], config)
    frame = features.merge(
        trainable[["show_id"]].assign(bucket=labels["bucket"].to_numpy(), rating=trainable["rating"].to_numpy()),
        on="show_id",
    )
    frame = frame.loc[frame["bucket"].notna()].reset_index(drop=True)
    return config, frame


def test_stratified_split_does_not_raise_on_the_fixture():
    _config, frame = _prepared()
    stratified_folds(frame["bucket"].astype(str), n_splits=5, seed=RANDOM_SEED)
    train_test_split(frame, frame["bucket"], test_size=0.15, stratify=frame["bucket"], random_state=RANDOM_SEED)


def test_predict_proba_rows_sum_to_one_and_cover_every_class():
    config, frame = _prepared()
    y = frame["bucket"].astype(str)
    preprocessor = make_preprocessor(config, use_description=description_has_text(frame), nonnegative=False)
    matrix = preprocessor.fit_transform(frame, y)
    estimator = make_estimator("logistic_regression", config)
    estimator.fit(matrix, y)
    probabilities = estimator.predict_proba(matrix)
    assert probabilities.shape[1] == y.nunique()
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)


def test_unseen_category_and_single_row_do_not_crash():
    config, frame = _prepared()
    y = frame["bucket"].astype(str)
    preprocessor = make_preprocessor(config, use_description=description_has_text(frame), nonnegative=False)
    preprocessor.fit(frame.iloc[:-1], y.iloc[:-1])
    holdout = frame.iloc[[-1]].copy()
    holdout["primary_country"] = "Atlantis"
    holdout["primary_genre"] = "Made Up Genre"
    holdout["listed_in"] = "Made Up Genre"
    transformed = preprocessor.transform(holdout)
    estimator = make_estimator("decision_tree", config)
    estimator.fit(preprocessor.transform(frame.iloc[:-1]), y.iloc[:-1])
    predicted = estimator.predict(transformed)
    assert len(predicted) == 1


def test_sentinel_token_in_the_holdout_is_absent_from_the_fitted_vocabulary():
    config, frame = _prepared()
    y = frame["bucket"].astype(str)
    train, holdout = train_test_split(frame, test_size=0.25, stratify=y, random_state=RANDOM_SEED)
    holdout = holdout.copy()
    holdout["title"] = holdout["title"] + " zzsentineltoken"
    preprocessor = make_preprocessor(config, use_description=description_has_text(train), nonnegative=False)
    preprocessor.fit(train, train["bucket"].astype(str))
    vocabulary = preprocessor.named_transformers_["title"].named_steps["tfidf"].vocabulary_
    assert "zzsentineltoken" not in vocabulary


def test_two_seeded_fits_match():
    config, frame = _prepared()
    y = frame["bucket"].astype(str)
    preprocessor = make_preprocessor(config, use_description=False, nonnegative=False)
    matrix = preprocessor.fit_transform(frame)
    first = make_estimator("decision_tree", config)
    second = make_estimator("decision_tree", config)
    first.fit(matrix, y)
    second.fit(matrix, y)
    assert np.array_equal(first.predict(matrix), second.predict(matrix))
