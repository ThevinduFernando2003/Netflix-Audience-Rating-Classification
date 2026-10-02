# Tuning results — Formulation B

The product label is the 4-level maturity bucket. This search does not score the sealed test set.

## Budget

- Search: `HalvingRandomSearchCV`, `40` candidates, factor `4`.
- Resource: training rows, from `400` up to `2000` (not the full pool).
- Ranking metric inside the search: macro-F1. Folds: `5`, seed `42`.
- After the search, each family's best parameter set is scored again with a 5-fold on **all** Formulation B training rows. Those rows are the numbers in the table below.
- Primary selection metric: full-pool macro-F1. QWK and balanced accuracy are reported beside it.
- Families searched: `logistic_regression`, `decision_tree`, `random_forest`, `hist_gradient_boosting`.
- Dummies, extra trees, and ComplementNB stay at the untuned comparison in `reports/target_design.md`. Extra trees tracked random forest there, and ComplementNB was behind on QWK.
- HistGradientBoosting `max_iter` in the search is 30 or 50 and `max_bins` is 32 or 64. A 255-bin histogram on the full title matrix did not finish a single formulation in 25 minutes during the comparison.
- Validation curves use a stratified subset of `2000` training rows and `3` folds, so the overfitting boundary can be drawn without another full search.
- Wall clock, search plus full-pool rescoring plus curves plus final fit: **834 seconds**.

## Search spaces

### `logistic_regression`

| parameter | values |
|---|---|
| `preprocess__title__tfidf__min_df` | 2, 3, 5 |
| `preprocess__title__tfidf__max_features` | 2000, 3500, 5000 |
| `preprocess__title__tfidf__ngram_range` | [1, 1], [1, 2] |
| `clf__C` | 0.03, 0.1, 0.3, 1.0, 3.0, 10.0 |

### `decision_tree`

| parameter | values |
|---|---|
| `preprocess__title__tfidf__min_df` | 2, 3, 5 |
| `preprocess__title__tfidf__max_features` | 2000, 3500, 5000 |
| `preprocess__title__tfidf__ngram_range` | [1, 1], [1, 2] |
| `clf__max_depth` | 6, 10, 14, 20, None |
| `clf__min_samples_split` | 2, 10, 20, 40 |
| `clf__min_samples_leaf` | 1, 5, 10, 20 |
| `clf__max_features` | sqrt, log2, 0.3 |
| `clf__criterion` | gini, entropy |
| `clf__ccp_alpha` | 0.0, 0.0005, 0.001, 0.005 |

### `random_forest`

| parameter | values |
|---|---|
| `preprocess__title__tfidf__min_df` | 2, 3, 5 |
| `preprocess__title__tfidf__max_features` | 2000, 3500, 5000 |
| `preprocess__title__tfidf__ngram_range` | [1, 1], [1, 2] |
| `clf__n_estimators` | 60, 120, 200 |
| `clf__max_depth` | 12, 18, None |
| `clf__min_samples_split` | 2, 8, 20 |
| `clf__min_samples_leaf` | 1, 2, 5 |
| `clf__max_features` | sqrt, 0.3 |
| `clf__criterion` | gini, entropy |

### `hist_gradient_boosting`

| parameter | values |
|---|---|
| `preprocess__title__tfidf__min_df` | 2, 3, 5 |
| `preprocess__title__tfidf__max_features` | 2000, 3500, 5000 |
| `preprocess__title__tfidf__ngram_range` | [1, 1], [1, 2] |
| `clf__learning_rate` | 0.05, 0.1, 0.15 |
| `clf__max_iter` | 30, 50 |
| `clf__max_leaf_nodes` | 15, 31, 63 |
| `clf__min_samples_leaf` | 10, 20, 40 |
| `clf__l2_regularization` | 0.0, 0.1, 1.0 |
| `clf__max_bins` | 32, 64 |

## Full training-pool 5-fold scores of each family's best parameters

| model | macro-F1 | balanced accuracy | QWK | train macro-F1 | halving macro-F1 | search seconds |
|---|---:|---:|---:|---:|---:|---:|
| `logistic_regression` | 0.5576 ± 0.0155 | 0.5861 ± 0.0214 | 0.6426 ± 0.0284 | 0.6462 ± 0.0072 | 0.5175 ± 0.0195 | 153 |
| `decision_tree` | 0.4938 ± 0.0223 | 0.5319 ± 0.0164 | 0.5556 ± 0.0353 | 0.5442 ± 0.0166 | 0.4334 ± 0.0248 | 24 |
| `random_forest` | 0.5977 ± 0.0139 | 0.5830 ± 0.0144 | 0.6596 ± 0.0178 | 0.9742 ± 0.0015 | 0.5476 ± 0.0334 | 166 |
| `hist_gradient_boosting` | 0.5768 ± 0.0130 | 0.5825 ± 0.0176 | 0.6620 ± 0.0229 | 0.8669 ± 0.0065 | 0.5257 ± 0.0363 | 152 |

## Selected model: `random_forest`

Chosen by full-pool macro-F1. The pipeline is refit on every Formulation B training row and written to `models/best_model.joblib`.

| parameter | value |
|---|---|
| `preprocess__title__tfidf__ngram_range` | `[1, 2]` |
| `preprocess__title__tfidf__min_df` | `2` |
| `preprocess__title__tfidf__max_features` | `5000` |
| `clf__n_estimators` | `200` |
| `clf__min_samples_split` | `8` |
| `clf__min_samples_leaf` | `1` |
| `clf__max_features` | `sqrt` |
| `clf__max_depth` | `None` |
| `clf__criterion` | `gini` |

## Halving rungs

### `logistic_regression`

| rung | candidates | rows | best macro-F1 on that rung |
|---:|---:|---:|---:|
| 0 | 40 | 400 | 0.4965 |
| 1 | 10 | 1600 | 0.5175 |

### `decision_tree`

| rung | candidates | rows | best macro-F1 on that rung |
|---:|---:|---:|---:|
| 0 | 40 | 400 | 0.4353 |
| 1 | 10 | 1600 | 0.4334 |

### `random_forest`

| rung | candidates | rows | best macro-F1 on that rung |
|---:|---:|---:|---:|
| 0 | 40 | 400 | 0.4864 |
| 1 | 10 | 1600 | 0.5476 |

### `hist_gradient_boosting`

| rung | candidates | rows | best macro-F1 on that rung |
|---:|---:|---:|---:|
| 0 | 40 | 400 | 0.4766 |
| 1 | 10 | 1600 | 0.5257 |

## Validation curves

Figure: `reports/figures/validation_curves_formulation_b.png`.

For max depth, the train/validation gap widens toward the flexible end of the grid (None). For trees, validation macro-F1 does not collapse at the flexible end of this grid; the gap stays about 0.431.

- `clf__max_depth`: widest train−validation gap is 0.429 macro-F1 at `None`.
- `clf__n_estimators`: widest train−validation gap is 0.431 macro-F1 at `120`.

## Re-run

```
python scripts/tune.py
python scripts/train.py
```

`scripts/train.py` refits the saved parameter set and does not repeat the search.
