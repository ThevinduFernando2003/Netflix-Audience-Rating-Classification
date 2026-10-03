# Netflix audience rating classification

Predict a title's maturity bucket. The product label is Formulation B, a 4-level ladder shared by the film and TV certificates: Kids, Older Kids, Teens, Mature.

The numbers below are copied from `reports/target_design.md`, `reports/tuning_results.md`, and `reports/test_evaluation.md`. Re-run the scripts named in those files to refresh them.

## Why this label

The catalogue mixes two certification systems, and the selection rule was pooled out-of-fold quadratic weighted kappa on the training pool, not accuracy. Logistic regression scored Formulation B at 0.6417, Formulation A at 0.5969, and Formulation C at 0.5792. Random forest agreed: A 0.5803, B 0.6049, C 0.5873. `NR` and `UR` are not on the ladder. Formulation B drops the 69 unrated rows in the training pool.

Training-pool counts after that drop:

| bucket | rows |
|---|---:|
| Mature | 3406 |
| Teens | 2250 |
| Older Kids | 1264 |
| Kids | 482 |

Description TF-IDF is omitted because that column is empty. The model uses the title, genres, and the other metadata that are actually present.

## Tuned model

The production model is a random forest, chosen by 5-fold macro-F1 on all Formulation B training rows. Validation macro-F1 was 0.5977 ± 0.0139. Training macro-F1 on those same folds was 0.9742 ± 0.0015. Quadratic weighted kappa was 0.6596 ± 0.0178. The search took 834 seconds and did not score the sealed test set.

Saved parameters: title TF-IDF unigrams and bigrams, `min_df` 2, 5,000 features, 200 trees, unlimited depth, `min_samples_split` 8, `min_samples_leaf` 1, `max_features` sqrt, gini, `class_weight=balanced_subsample`. The fitted pipeline is `models/best_model.joblib` (not in git).

## Holdout

The sealed split has 1,319 titles. 13 unrated titles are excluded. **1,306** labelled titles were scored once. Those scores were not used to change parameters.

| metric | holdout | same fitted model on its training rows |
|---|---:|---:|
| accuracy | 0.6363 | 0.9722 |
| macro-F1 | 0.5973 | 0.9714 |
| balanced accuracy | 0.5776 | 0.9744 |
| quadratic weighted kappa | 0.6546 | 0.9805 |
| ordinal MAE | 0.4280 | 0.0299 |
| adjacent accuracy | 0.9449 | 0.9980 |
| severe-error rate | 0.0551 | 0.0020 |
| Mature predicted as Kids | 4 | 0 |
| Kids predicted as Mature | 8 | 0 |

Training macro-F1 on the rows the forest was fit on is 0.9714, against a holdout macro-F1 of 0.5973, so the forest memorizes titles it has already seen. Holdout macro-F1 matches the cross-validation estimate of 0.5977 ± 0.0139. Top-label expected calibration error on the holdout is 0.0691. Mature predicted as Kids, the safety-critical error, happened 4 times. Confusion matrices and the reliability curve are in `reports/figures/`.

A pre-specified unweighted forest was also scored and was not adopted. On the training folds it raised accuracy to 0.6348 and lowered macro-F1 to 0.5765. A soft vote and a short list of class-probability scales were checked the same way. None beat the balanced forest on both accuracy and macro-F1. See `reports/improvement_cv.md`.

## Re-run

```text
python scripts/profile_data.py
python scripts/design_target.py
python scripts/tune.py
python scripts/evaluate_test.py
```

`scripts/evaluate_test.py` scores the saved model. It does not search again.
