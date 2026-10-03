# Summary

This project predicts a Netflix title's maturity bucket. The product label is Formulation B: one 4-level ladder shared by the film and TV certificates. Kids, Older Kids, Teens, and Mature are the classes, in that order. The fitted model is a random forest saved at `models/best_model.joblib`. It is not in git.

Every figure below comes from `reports/target_design.md`, `reports/tuning_results.md`, `reports/test_evaluation.md`, or `reports/improvement_cv.md`. The reasoning behind each non-obvious choice is in `docs/DECISIONS.md`. Methods are in `docs/METHODS.md`, the result tables in `docs/FINDINGS.md`, and the code layout in `docs/ARCHITECTURE.md`.

## What the file actually is

The catalogue used here has 8,790 rows and 10 columns. `cast` and `description` are absent. Description TF-IDF is not fitted, because that column is empty. `director` and `country` use the sentinel `Not Given` (2,588 directors, 287 countries). That sentinel is treated as missing. Most movies carry a TV certificate: 4,430 of 6,126 movies, which is 72.3%. A film-certificate-only movie model would drop the majority of movies.

## Which label was chosen

Three label spaces were compared on the training pool only, with seed 42 and a sealed 15% holdout left unscored. The pre-registered rule was logistic-regression pooled quadratic weighted kappa, confirmed by random forest within 0.02.

| model | A | B | C |
|---|---:|---:|---:|
| logistic regression | 0.5969 | 0.6417 | 0.5792 |
| random forest | 0.5803 | 0.6049 | 0.5873 |

Formulation B won both. On that ladder, logistic regression also had ordinal MAE 0.4789, adjacent accuracy 0.9287, and 38 Mature-as-Kids errors, against 117 for A and 125 for C. Training-pool counts after dropping 69 unrated rows are Mature 3,406, Teens 2,250, Older Kids 1,264, and Kids 482.

## What was trained

Four families were tuned with `HalvingRandomSearchCV` (40 candidates, at most 2,000 rows per rung), then each winner was rescored with a real 5-fold on all 7,402 Formulation B training rows. Selection was full-pool macro-F1. The random forest won: validation macro-F1 0.5977 ± 0.0139, training macro-F1 on those folds 0.9742 ± 0.0015, quadratic weighted kappa 0.6596 ± 0.0178. The search, rescoring, and validation curves took 834 seconds.

The saved forest uses title TF-IDF unigrams and bigrams, `min_df` 2, 5,000 features, 200 trees, unlimited depth, `min_samples_split` 8, `min_samples_leaf` 1, `max_features` sqrt, gini, and `class_weight=balanced_subsample`.

## What the holdout says

The sealed split has 1,319 titles. 13 unrated titles are excluded. 1,306 labelled titles were scored once. Those scores did not change any parameter.

| metric | holdout | same model on its training rows |
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

Training macro-F1 on the rows the forest was fit on is 0.9714, against a holdout macro-F1 of 0.5973. The forest memorizes titles it has already seen. The holdout macro-F1 matches the cross-validation estimate of 0.5977 ± 0.0139, so the gap was already visible before the holdout was opened. Top-label expected calibration error is 0.0691. Most mistakes are one step on the ladder. Mature predicted as Kids happened 4 times out of 601 Mature holdout titles.

## What was tried afterwards, and not adopted

A soft vote of the three tuned models, eight class-probability scales, and a `class_weight=None` forest were all scored on training folds only. None beat the balanced forest on both accuracy and macro-F1. The highest-macro-F1 vote (logistic regression 1, forest 2, boosting 1) reached macro-F1 0.5993 and quadratic weighted kappa 0.6780, with accuracy 0.6298 against the forest's 0.6308. The saved model was not replaced, and the holdout was not scored again to choose a model.

An external language-model rating API was not used. The catalogue is public, so a model can recite the certificate. That would leak the label.

## How to read a new title

`POST /predict` and `app/streamlit_app.py` load the saved pipeline. They return the four buckets in ordinal order. A Kids prediction shows the severe-error warning, because Mature predicted as Kids is the safety-critical failure. Docker Compose mounts the local `models` directory. GitHub Actions runs the tests and does not need the fitted file.
