"""Fold-fitted preprocessing. Vocabulary, medians, and category levels are learned in fit only."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MultiLabelBinarizer, OneHotEncoder, StandardScaler

from src.features.engineer import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def _as_text(values: pd.DataFrame | np.ndarray) -> list[str]:
    if isinstance(values, pd.DataFrame):
        series = values.iloc[:, 0]
    else:
        array = np.asarray(values)
        series = array[:, 0] if array.ndim == 2 else array
    output: list[str] = []
    for value in series:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            output.append("")
        else:
            output.append(str(value))
    return output


def _split_genres(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


class GenreBinarizer(BaseEstimator, TransformerMixin):
    """One binary column per genre seen in fit. Unseen genres become a zero row, not an error."""

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> GenreBinarizer:
        lists = [_split_genres(text) for text in _as_text(X)]
        self.mlb_ = MultiLabelBinarizer()
        self.mlb_.fit(lists)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        lists = [_split_genres(text) for text in _as_text(X)]
        return self.mlb_.transform(lists).astype(np.float32)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray([f"genre__{name}" for name in self.mlb_.classes_], dtype=object)


class ToDense(BaseEstimator, TransformerMixin):
    """HistGradientBoosting needs a dense block. Fit learns nothing."""

    def fit(self, X: Any, y: Any = None) -> ToDense:
        return self

    def transform(self, X: Any) -> np.ndarray:
        if hasattr(X, "toarray"):
            return X.toarray().astype(np.float32, copy=False)
        return np.asarray(X, dtype=np.float32)


class ClipAtZero(BaseEstimator, TransformerMixin):
    """Make a numeric block safe for ComplementNB. Fit learns nothing."""

    def fit(self, X: Any, y: Any = None) -> ClipAtZero:
        return self

    def transform(self, X: Any) -> np.ndarray:
        array = np.asarray(X, dtype=np.float64)
        return np.maximum(array, 0.0)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is None:
            raise ValueError("ClipAtZero.get_feature_names_out requires input_features.")
        return np.asarray(input_features)


def _text_pipeline(vectorizer: TfidfVectorizer) -> Pipeline:
    """ColumnTransformer hands a one-column frame to each branch. TfidfVectorizer needs a 1d list of strings."""
    return Pipeline(
        [
            ("flatten", FunctionTransformer(_as_text, validate=False)),
            ("tfidf", vectorizer),
        ]
    )


def _title_vectorizer(config: dict[str, Any]) -> TfidfVectorizer:
    params = config["features"]["title_tfidf"]
    ngram = tuple(int(item) for item in params["ngram_range"])
    return TfidfVectorizer(
        ngram_range=ngram,  # type: ignore[arg-type]
        min_df=int(params["min_df"]),
        max_features=int(params["max_features"]),
        strip_accents=params["strip_accents"],
        dtype=np.float32,
    )


def _description_vectorizer(config: dict[str, Any]) -> TfidfVectorizer:
    params = config["features"]["description_tfidf"]
    ngram = tuple(int(item) for item in params["ngram_range"])
    return TfidfVectorizer(
        ngram_range=ngram,  # type: ignore[arg-type]
        min_df=int(params["min_df"]),
        max_features=int(params["max_features"]),
        sublinear_tf=bool(params["sublinear_tf"]),
        stop_words=params["stop_words"],
        strip_accents=params["strip_accents"],
        dtype=np.float32,
    )


def _numeric_pipeline(nonnegative: bool) -> Pipeline:
    steps: list[tuple[str, BaseEstimator]] = [
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if nonnegative:
        steps.append(("clip", ClipAtZero()))
    else:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def make_preprocessor(config: dict[str, Any], *, use_description: bool, nonnegative: bool) -> ColumnTransformer:
    """ColumnTransformer fitted inside each CV fold. `nonnegative` selects the ComplementNB branch."""
    min_frequency = int(config["features"]["categorical"]["min_frequency"])
    transformers: list[tuple[str, BaseEstimator, list[str]]] = [
        ("title", _text_pipeline(_title_vectorizer(config)), ["title"]),
        ("genres", GenreBinarizer(), ["listed_in"]),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=min_frequency,
                sparse_output=True,
                dtype=np.float32,
            ),
            CATEGORICAL_COLUMNS,
        ),
        ("numeric", _numeric_pipeline(nonnegative), NUMERIC_COLUMNS),
    ]
    if use_description:
        transformers.insert(1, ("description", _text_pipeline(_description_vectorizer(config)), ["description"]))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)
